import ast
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from core.openheaders import get_segy_readable_text

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SEGYHeaderParser:
    """
    Parses SEG-Y textual headers to extract attribute byte locations and revision/format metadata via LLM.
    """

    MODEL_NAME = "meta-llama/Meta-Llama-3.3-70B-Instruct"  # Default model name

    INITIAL_ATTR_PROMPT = (
        "You are a senior geophysicist and data-extraction engine.\n"
        "Given only the SEG-Y textual header below, identify every metadata field STRICTLY expicitly found ONLY in the Textual header PROVIDED, do not get attributes from standard segy formats\n"
        "whose byte locations are explicitly stated or clearly inferred from the information in the SEG-Y textual header.\n"
        "Respond strictly with a JSON list of dictionaries in this schema:\n"
        "[{'attribute': 'Exact field name as in textual header provided', 'bytelocation': 'startbyte–endbyte', 'confidence': 'high|medium|low'}, …]\n\n"
        " Example of inferred information: when bytelocation is only a number like 133, it is assumed to be the startbyte, but then context says it is a bytesize of 32bit = 4 byte, implying endbyte = startbyte + 4byte - 1byte"
        " Or when the bytesize is absent or non inferrable, and startbyte is found, the field 'bytelocation': 'startbyte-None'"
        " remember to ignore the line labels usually at start of each line, like C10, C 9, C09 etc."
        "Rules:\n"
        "- confidence 'high': header explicitly gives the byte range\n"
        "- confidence 'medium': clear inference from context \n"
        "- confidence 'low': ambiguous label or approximate guess\n"
        "- If repeated attributes found, include all of them in the list of dictionaries, try your best to add 'attribute': attribute(context) to give more information about the attribute how it differs from each other"
        "- Omit any field you cannot ground in the text\n"
        "- Normalize byte ranges to 'startbyte–endbyte' format\n"
        "- No prose—only raw JSON"
    )

    CRITIC_ATTR_PROMPT = (
        "You are a strict validator.\n"
        "Given the same SEG-Y textual header and the JSON output from the extractor,\n"
        "review each entry:\n"
        "- Confirm the attribute name matches verbatim to SEG-Y textual header\n"
        "- Unless it is a repeated attribute then there could be a field attribute(contex) to give context, ensure this context is relevant to the SEG-Y textual header, and ensure the first part of the attribute name has to be verbatim to provided textual header"
        "- Confirm the byte range is explicitly stated or strongly inferable\n"
        " Example of inferred information: when only startbyte is found, but then context says it is a bytesize of 32bit = 4 byte, implying endbyte = startbyte + 4byte - 1byte"
        " Or when the bytesize is absent or non inferrable, and startbyte is found, the field 'bytelocation': 'startbyte-None'"
        " remember to ignore the line labels usually at start of each line, like C10, C 9, C09 etc."
        "- Downgrade confidence if only partially supported\n"
        "- Remove entries with no support in the header\n\n"
        "Respond ONLY with the corrected JSON list in the same schema—no explanations."
    )

    INITIAL_REVISION_PROMPT = (
        "You are a SEG-Y expert.\n"
        "From the textual header, detect the format revision/version.\n"
        "Respond with exactly one JSON dict:\n"
        "{'revision':'0|0.0|1|1.0|2|2.0|2.1|None','confidence':'high|medium|low'}\n\n"
        " remember to ignore the line labels usually at start of each line, like C10, C 9, C09 etc."
        "Rules:\n"
        "- 'high': exact phrase like 'SEG-Y revision 1.0' appears\n"
        "- 'medium': strong contextual clues\n"
        "- 'low': weak or conflicting hints\n"
        "- If no evidence, output {'revision':'None','confidence':'high'}\n"
        "- No extra text—only the JSON dict"
    )

    CRITIC_REVISION_PROMPT = (
        "You are a disciplined QA reviewer.\n"
        "Given the textual header and previous JSON dict,\n"
        "verify the revision value against the header:\n"
        "- Must be one of [0,0.0,1,1.0,2,2.0,2.1,None]\n"
        "- Adjust confidence to match evidence level\n"
        "- If hallucinated or unsupported, set {'revision':'None','confidence':'medium'}\n"
        "- If header has no revision and previous was non-None, override to\n"
        "  {'revision':'None','confidence':'high'}\n\n"
        " remember to ignore the line labels usually at start of each line, like C10, C 9, C09 etc."
        "Respond ONLY with the finalized JSON dict—no commentary."
    )





    def __init__(
        self,
        server_url: str,
        temperature: float = 0.3,
        max_tokens: int = 20000,
        verbose: bool = False,
    ) -> None:
        self.server_url = server_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

    def _build_payload(self, system_prompt: str, user_content: str) -> Dict:
        return {
            "model": self.MODEL_NAME,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

    def _post(self, payload: Dict) -> Dict:
        resp = requests.post(self.server_url, json=payload, verify=False)
        resp.raise_for_status()
        return resp.json()

    def _parse_json_array(self, content: str) -> List[Dict[str, str]]:
        """Parse JSON array using ast.literal_eval."""
        print(content)
        try:
            return ast.literal_eval(content.strip())
        except Exception as e:
            logging.error(f"JSON parse failed: {e}")
            return []

    def _validate_array(self, entries: List[Dict[str, str]]) -> bool:
        """Validate that entries have required keys and valid confidence levels."""
        if not isinstance(entries, list):
            return False
            
        for item in entries:
            if not isinstance(item, dict):
                return False
                
            # Check required keys
            if "attribute" not in item or "bytelocation" not in item or "confidence" not in item:
                return False
                
            # Check confidence is valid
            if item["confidence"] not in ['high', 'medium', 'low']:
                return False
                
        return True

    def get_summary(self, attributes: List[Dict[str, str]], revision_info: Dict[str, str]) -> str:
        """Generate a summary of the parsing results."""
        summary = []
        
        # Revision summary
        rev_val = revision_info.get("revision", "None")
        rev_conf = revision_info.get("confidence", "low")
        if rev_val != "None":
            summary.append(f"SEG-Y Revision: {rev_val} [{rev_conf}]")
        else:
            summary.append(f"SEG-Y Revision: Not detected [{rev_conf}]")
            
        summary.append(f"Attributes found: {len(attributes)}")
        
        if attributes:
            # Count by confidence level
            confidence_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
            for attr in attributes:
                conf = attr.get("confidence", "unknown")
                confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
            
            summary.append("Confidence breakdown:")
            for conf, count in confidence_counts.items():
                if count > 0:
                    summary.append(f"  {conf}: {count}")
            
            summary.append("Attribute-byte location pairs:")
            for i, attr in enumerate(attributes, 1):
                conf_str = f" [{attr.get('confidence', 'unknown')}]"
                summary.append(f"  {i}. {attr['attribute']} -> {attr['bytelocation']}{conf_str}")
        
        return "\n".join(summary)

    def nlp_extract_attribute_bytelocation(
        self,
        header_text: str,
        retries: int = 2,
        critic_cycles: int = 1,
        delay: float = 0.5,
    ) -> List[Dict[str, str]]:
        """
        Extract attributes with byte locations via LLM with validation cycles.
        """
        entries: List[Dict[str, str]] = []
        
        # Initial extraction
        payload = self._build_payload(self.INITIAL_ATTR_PROMPT, header_text)
        
        for cycle in range(critic_cycles + 1):
            if self.verbose:
                logging.info(f"Attribute extraction cycle {cycle + 1}/{critic_cycles + 1}")

            for attempt in range(retries + 1):
                try:
                    resp = self._post(payload)
                    content = resp["choices"][0]["message"]["content"].strip()
                    
                    # Clean up response - remove markdown formatting if present
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "").strip()
                    elif content.startswith("```"):
                        content = content.replace("```", "").strip()
                    
                    if self.verbose:
                        logging.debug(f"Raw LLM response: {content}")
                    
                    entries = self._parse_json_array(content)
                    if self.verbose and not entries:
                        logging.debug(f"JSON parsing failed for: {content[:200]}...")
                    # Standardize byte location format
                    entries = self._standardize_byte_locations(entries)
                    if self._validate_array(entries):
                        if self.verbose:
                            logging.info(f"Successfully extracted {len(entries)} entries")
                        break
                        
                except Exception as e:
                    logging.error(f"Extraction attempt {attempt + 1} failed: {e}")
                    if attempt < retries:
                        time.sleep(delay)

            # Apply critic validation if we have entries and more cycles to go
            if cycle < critic_cycles and entries:
                critic_input = f"Original header:\n{header_text}\n\nExtracted entries:\n{entries}"
                payload = self._build_payload(self.CRITIC_ATTR_PROMPT, critic_input)
                time.sleep(delay)

        return entries

    def nlp_detect_revision(
        self,
        header_text: str,
        retries: int = 2,
        critic_cycles: int = 1,
        delay: float = 0.5,
    ) -> Dict[str, str]:
        """
        Extract revision/version via LLM with validation.
        Returns dict with 'revision' and 'confidence' keys.
        """
        revision_result = {"revision": "None", "confidence": "low"}
        payload = self._build_payload(self.INITIAL_REVISION_PROMPT, header_text)

        for cycle in range(critic_cycles + 1):
            if self.verbose:
                logging.info(f"Revision detection cycle {cycle + 1}/{critic_cycles + 1}")

            for attempt in range(retries + 1):
                try:
                    resp = self._post(payload)
                    content = resp["choices"][0]["message"]["content"].strip()
                    
                    # Clean up response - remove markdown formatting if present
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "").strip()
                    elif content.startswith("```"):
                        content = content.replace("```", "").strip()
                    
                    # Parse the JSON response
                    parsed_result = self._parse_revision_json(content)
                    if parsed_result:
                        revision_result = parsed_result
                        if self.verbose:
                            logging.info(f"Detected revision: {revision_result}")
                        break
                        
                except Exception as e:
                    logging.error(f"Revision detection attempt {attempt + 1} failed: {e}")
                    if attempt < retries:
                        time.sleep(delay)

            # Apply critic validation
            if cycle < critic_cycles:
                critic_input = f"Header text:\n{header_text}\n\nPrevious response: {revision_result}"
                payload = self._build_payload(self.CRITIC_REVISION_PROMPT, critic_input)
                time.sleep(delay)

        return revision_result

    def _parse_revision_json(self, content: str) -> Optional[Dict[str, str]]:
        """Parse revision JSON using ast.literal_eval."""
        try:
            result = ast.literal_eval(content.strip())
            if isinstance(result, dict) and "revision" in result and "confidence" in result:
                return self._normalize_confidence(result)
        except Exception as e:
            logging.error(f"Failed to parse revision JSON: {e}")
        return None

    def _normalize_confidence(self, result: Dict[str, str]) -> Dict[str, str]:
        """Normalize confidence levels to expected values."""
        confidence_map = {
            'uncertain': 'low',
            'weak': 'low', 
            'poor': 'low',
            'good': 'medium',
            'strong': 'high',
            'very high': 'high',
            'explicit': 'high'
        }
        
        conf = result.get('confidence', 'low').lower().strip()
        if conf in confidence_map:
            result['confidence'] = confidence_map[conf]
        elif conf not in ['high', 'medium', 'low']:
            result['confidence'] = 'low'
            
        return result

    def _standardize_byte_locations(self, entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Standardize byte location format to 'start-end'."""
        import re
        
        standardized = []
        for entry in entries:
            if "bytelocation" in entry:
                byte_loc = entry["bytelocation"]
                
                # Extract numbers from various formats
                # Patterns: "13-16", "13:16", "13 to 16", "bytes 13-16", "13- 16", etc.
                numbers = re.findall(r'\d+', byte_loc)
                
                if len(numbers) >= 2:
                    start_byte = numbers[0]
                    end_byte = numbers[1]
                    entry["bytelocation"] = f"{start_byte}-{end_byte}"
                elif len(numbers) == 1:
                    # Single number - might be a size, keep as is but log warning
                    logging.warning(f"Single number in byte location: {byte_loc}")
                
            standardized.append(entry)
        
        return standardized

    def parse_text_header(self, header_text: str) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """
        Parse SEG-Y header text and return attributes and revision separately.
        
        Returns:
            Tuple of (attributes_list, revision_dict)
            - attributes_list: List of dicts with 'attribute', 'bytelocation', 'confidence'
            - revision_dict: Dict with 'revision' and 'confidence' keys
        """
        attrs = self.nlp_extract_attribute_bytelocation(header_text)
        revision_info = self.nlp_detect_revision(header_text)
        
        return attrs, revision_info

    def process_segy_file(self, segy_path: Path) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """
        Process SEG-Y file and return attributes and revision separately.
        
        Returns:
            Tuple of (attributes_list, revision_dict)
            - attributes_list: List of dicts with 'attribute', 'bytelocation', 'confidence'
            - revision_dict: Dict with 'revision' and 'confidence' keys
        """
        header_text = get_segy_readable_text(segy_path, verbose=self.verbose)
        return self.parse_text_header(header_text)


if __name__ == "__main__":
    import sys
    from core.utils import import_segys, DEFAULT_SEGYS

    if len(sys.argv) != 2:
        print("Usage: python script.py <index>")
        sys.exit(1)

    index = int(sys.argv[1])
    segy_file = Path(import_segys(DEFAULT_SEGYS)[index])
    
    print(f"Processing: {segy_file}")
    
    # Use environment variables for LLM configuration
    import os
    server_url = os.getenv('LOCAL_LLM_SERVER_URL')
    
    if not server_url:
        print("❌ Error: LOCAL_LLM_SERVER_URL not set in environment variables")
        print("Please configure your .env file with LLM settings")
        sys.exit(1)
    
    parser = SEGYHeaderParser(
        server_url=server_url,
        verbose=True,
    )
    
    attributes, revision_info = parser.process_segy_file(segy_file)
    
    # Optional: Print summary
    print(f"\n{'-'*50}")
    print("SUMMARY:")
    print(parser.get_summary(attributes, revision_info))








