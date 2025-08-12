"""
Abstract LLM Provider interface and implementations for the AI-Powered SEGY Metadata Parser.

This module provides a unified interface for different LLM providers (Local LLMs, Gemini)
with fallback mechanisms and configuration management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import requests
import json
import os
import time
from pathlib import Path


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    success: bool
    error_message: Optional[str] = None
    provider_name: Optional[str] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def invoke_prompt(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        """
        Invoke a prompt with the LLM provider.
        
        Args:
            system_prompt: System/context prompt
            user_prompt: User query/prompt
            **kwargs: Provider-specific parameters
            
        Returns:
            LLMResponse with the result
        """
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the provider with new settings."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        pass


class LlamaProvider(LLMProvider):
    """Local LLM provider for OpenAI-compatible API endpoints."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server_url = config.get('server_url') or config.get('base_url')
        self.model = config.get('model', 'meta-llama/Meta-Llama-3.3-70B-Instruct')
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 20000)
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
    
    def _build_payload(self, system_prompt: str, user_content: str) -> Dict:
        """Build OpenAI-compatible API payload."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
    
    def _post(self, payload: Dict) -> Dict:
        """Make HTTP POST request to LLM API."""
        resp = requests.post(self.server_url, json=payload, verify=False)
        resp.raise_for_status()
        return resp.json()
    
    def invoke_prompt(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        """Invoke local LLM via OpenAI-compatible API."""
        start_time = time.time()
        
        # Override temperature and max_tokens if provided in kwargs
        original_temp = self.temperature
        original_max_tokens = self.max_tokens
        
        if 'temperature' in kwargs:
            self.temperature = kwargs['temperature']
        if 'max_tokens' in kwargs:
            self.max_tokens = kwargs['max_tokens']
        
        try:
            # Build API payload
            payload = self._build_payload(system_prompt, user_prompt)
            
            for attempt in range(self.max_retries):
                try:
                    # Make API request
                    resp = self._post(payload)
                    
                    # Extract content from response
                    content = resp["choices"][0]["message"]["content"].strip()
                    
                    # Clean up response - remove markdown formatting if present
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "").strip()
                    elif content.startswith("```"):
                        content = content.replace("```", "").strip()
                    
                    response_time = time.time() - start_time
                    
                    return LLMResponse(
                        content=content,
                        success=True,
                        provider_name=self.name,
                        response_time=response_time
                    )
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        return LLMResponse(
                            content="",
                            success=False,
                            error_message=f"Request failed after {self.max_retries} attempts: {str(e)}",
                            provider_name=self.name
                        )
                    time.sleep(0.5)  # Brief delay between retries
            
            return LLMResponse(
                content="",
                success=False,
                error_message="Max retries exceeded",
                provider_name=self.name
            )
            
        finally:
            # Restore original values
            self.temperature = original_temp
            self.max_tokens = original_max_tokens
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Update configuration."""
        self.config.update(config)
        self.base_url = config.get('base_url', self.base_url)
        self.model = config.get('model', self.model)
        self.timeout = config.get('timeout', self.timeout)
        self.max_retries = config.get('max_retries', self.max_retries)
    
    def is_available(self) -> bool:
        """Check if local LLM service is available."""
        try:
            # For OpenAI-compatible endpoints, try a simple test request
            test_payload = {
                "model": self.model,
                "temperature": 0.1,
                "max_tokens": 10,
                "messages": [
                    {"role": "system", "content": "You are a test."},
                    {"role": "user", "content": "Hi"}
                ]
            }
            
            response = requests.post(
                self.server_url, 
                json=test_payload, 
                timeout=5,
                verify=False
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class GeminiProvider(LLMProvider):
    """Gemini provider using Google's Gemini API."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('GEMINI_API_KEY')
        self.model = config.get('model', 'gemini-pro')
        self.base_url = config.get('base_url', 'https://generativelanguage.googleapis.com/v1beta')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
    
    def invoke_prompt(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        """Invoke Gemini model via REST API."""
        if not self.api_key:
            return LLMResponse(
                content="",
                success=False,
                error_message="Gemini API key not configured",
                provider_name=self.name
            )
        
        start_time = time.time()
        
        # Gemini API format
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"System: {system_prompt}\n\nUser: {user_prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": kwargs.get('temperature', 0.7),
                "topP": kwargs.get('top_p', 0.9),
                "maxOutputTokens": kwargs.get('max_tokens', 4000)
            }
        }
        
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_time = time.time() - start_time
                    
                    # Extract content from Gemini response format
                    content = ""
                    if 'candidates' in result and len(result['candidates']) > 0:
                        candidate = result['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            parts = candidate['content']['parts']
                            if len(parts) > 0 and 'text' in parts[0]:
                                content = parts[0]['text']
                    
                    return LLMResponse(
                        content=content,
                        success=True,
                        provider_name=self.name,
                        response_time=response_time
                    )
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt == self.max_retries - 1:
                        return LLMResponse(
                            content="",
                            success=False,
                            error_message=error_msg,
                            provider_name=self.name
                        )
                    time.sleep(2 ** attempt)
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {str(e)}"
                if attempt == self.max_retries - 1:
                    return LLMResponse(
                        content="",
                        success=False,
                        error_message=error_msg,
                        provider_name=self.name
                    )
                time.sleep(2 ** attempt)
        
        return LLMResponse(
            content="",
            success=False,
            error_message="Max retries exceeded",
            provider_name=self.name
        )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Update configuration."""
        self.config.update(config)
        self.api_key = config.get('api_key', self.api_key)
        self.model = config.get('model', self.model)
        self.base_url = config.get('base_url', self.base_url)
        self.timeout = config.get('timeout', self.timeout)
        self.max_retries = config.get('max_retries', self.max_retries)
    
    def is_available(self) -> bool:
        """Check if Gemini API is available and configured."""
        if not self.api_key:
            return False
        
        try:
            # Test with a simple request
            test_payload = {
                "contents": [{"parts": [{"text": "Hello"}]}],
                "generationConfig": {"maxOutputTokens": 10}
            }
            
            url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
            response = requests.post(
                url,
                json=test_payload,
                timeout=5,
                headers={'Content-Type': 'application/json'}
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class LLMFactory:
    """Factory for creating and managing LLM provider instances."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.providers = {}
        self._initialize_providers()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from .env file and environment variables."""
        config = {
            'default_provider': os.getenv('DEFAULT_LLM_PROVIDER', 'local'),
            'fallback_providers': os.getenv('FALLBACK_LLM_PROVIDERS', 'gemini').split(','),
            'local': {
                'server_url': os.getenv('LOCAL_LLM_SERVER_URL'),
                'model': os.getenv('LOCAL_LLM_MODEL', 'meta-llama/Meta-Llama-3.3-70B-Instruct'),
                'temperature': float(os.getenv('LOCAL_LLM_TEMPERATURE', '0.3')),
                'max_tokens': int(os.getenv('LOCAL_LLM_MAX_TOKENS', '20000')),
                'timeout': int(os.getenv('LOCAL_LLM_TIMEOUT', '30')),
                'max_retries': int(os.getenv('LOCAL_LLM_MAX_RETRIES', '3'))
            },
            'gemini': {
                'api_key': os.getenv('GEMINI_API_KEY'),
                'model': os.getenv('GEMINI_MODEL', 'gemini-pro'),
                'timeout': int(os.getenv('GEMINI_TIMEOUT', '30')),
                'max_retries': int(os.getenv('GEMINI_MAX_RETRIES', '3'))
            }
        }
        
        # Load from config file if provided
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file {config_path}: {e}")
        
        return config
    
    def _initialize_providers(self):
        """Initialize all configured providers."""
        if 'local' in self.config:
            self.providers['local'] = LlamaProvider(self.config['local'])

        
        if 'gemini' in self.config:
            self.providers['gemini'] = GeminiProvider(self.config['gemini'])
    
    def create_provider(self, provider_name: str) -> Optional[LLMProvider]:
        """Create a specific provider instance."""
        if provider_name in self.providers:
            return self.providers[provider_name]
        return None
    
    def get_default_provider(self) -> Optional[LLMProvider]:
        """Get the configured default provider."""
        default_name = self.config.get('default_provider', 'local')
        return self.create_provider(default_name)
    
    def get_fallback_providers(self) -> List[LLMProvider]:
        """Get list of fallback providers in order."""
        fallback_names = self.config.get('fallback_providers', [])
        if isinstance(fallback_names, str):
            fallback_names = [name.strip() for name in fallback_names.split(',')]
        
        providers = []
        for name in fallback_names:
            provider = self.create_provider(name)
            if provider and provider.is_available():
                providers.append(provider)
        
        return providers
    
    def get_available_provider(self) -> Optional[LLMProvider]:
        """Get the first available provider (default or fallback)."""
        # Try default provider first
        default = self.get_default_provider()
        if default and default.is_available():
            return default
        
        # Try fallback providers
        for provider in self.get_fallback_providers():
            if provider.is_available():
                return provider
        
        return None


@dataclass
class ReasoningStep:
    """A single step in chain-of-thought reasoning."""
    step_number: int
    question: str
    reasoning: str
    conclusion: str
    confidence: float


@dataclass
class ReasoningResult:
    """Result of chain-of-thought reasoning process."""
    conclusion: str
    reasoning_steps: List[ReasoningStep]
    confidence: float
    alternative_interpretations: List[str]
    requires_validation: bool


class ChainOfThoughtReasoner:
    """Implements sophisticated chain-of-thought reasoning for complex parsing scenarios."""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.max_reasoning_steps = 5
        self.max_tokens_per_step = 4000
    
    def reason_through_ambiguity(self, context: str, hypotheses: List[str]) -> ReasoningResult:
        """
        Use multi-step reasoning to resolve ambiguous parsing scenarios.
        
        Args:
            context: The ambiguous textual header content or parsing context
            hypotheses: List of possible interpretations to evaluate
            
        Returns:
            ReasoningResult with step-by-step reasoning and conclusion
        """
        provider = self.llm_factory.get_available_provider()
        if not provider:
            return ReasoningResult(
                conclusion="No LLM provider available",
                reasoning_steps=[],
                confidence=0.0,
                alternative_interpretations=[],
                requires_validation=True
            )
        
        reasoning_steps = []
        current_context = context
        
        # Step 1: Analyze the ambiguous content
        step1_prompt = self._create_analysis_prompt(context, hypotheses)
        step1_response = provider.invoke_prompt(
            "You are an expert in SEGY file format analysis. Use systematic reasoning to analyze ambiguous header information.",
            step1_prompt,
            max_tokens=self.max_tokens_per_step
        )
        
        if step1_response.success:
            step1 = ReasoningStep(
                step_number=1,
                question="What information can be extracted from this header content?",
                reasoning=step1_response.content,
                conclusion=self._extract_conclusion(step1_response.content),
                confidence=self._estimate_confidence(step1_response.content)
            )
            reasoning_steps.append(step1)
            current_context += f"\n\nStep 1 Analysis: {step1.conclusion}"
        
        # Step 2: Evaluate each hypothesis
        step2_prompt = self._create_hypothesis_evaluation_prompt(current_context, hypotheses)
        step2_response = provider.invoke_prompt(
            "You are evaluating SEGY header parsing hypotheses. Systematically assess each possibility.",
            step2_prompt,
            max_tokens=self.max_tokens_per_step
        )
        
        if step2_response.success:
            step2 = ReasoningStep(
                step_number=2,
                question="Which hypotheses are most likely correct and why?",
                reasoning=step2_response.content,
                conclusion=self._extract_conclusion(step2_response.content),
                confidence=self._estimate_confidence(step2_response.content)
            )
            reasoning_steps.append(step2)
            current_context += f"\n\nStep 2 Evaluation: {step2.conclusion}"
        
        # Step 3: Consider SEGY standards and context
        step3_prompt = self._create_standards_prompt(current_context)
        step3_response = provider.invoke_prompt(
            "You are applying SEGY format standards to validate parsing decisions.",
            step3_prompt,
            max_tokens=self.max_tokens_per_step
        )
        
        if step3_response.success:
            step3 = ReasoningStep(
                step_number=3,
                question="How do SEGY standards inform this decision?",
                reasoning=step3_response.content,
                conclusion=self._extract_conclusion(step3_response.content),
                confidence=self._estimate_confidence(step3_response.content)
            )
            reasoning_steps.append(step3)
            current_context += f"\n\nStep 3 Standards: {step3.conclusion}"
        
        # Final synthesis
        final_prompt = self._create_synthesis_prompt(current_context, reasoning_steps)
        final_response = provider.invoke_prompt(
            "You are synthesizing chain-of-thought reasoning to reach a final conclusion about SEGY header parsing.",
            final_prompt,
            max_tokens=self.max_tokens_per_step
        )
        
        if final_response.success:
            final_conclusion = self._extract_conclusion(final_response.content)
            final_confidence = self._estimate_confidence(final_response.content)
            alternatives = self._extract_alternatives(final_response.content)
            requires_validation = self._requires_validation(final_response.content)
        else:
            final_conclusion = "Unable to reach conclusion due to LLM failure"
            final_confidence = 0.0
            alternatives = []
            requires_validation = True
        
        return ReasoningResult(
            conclusion=final_conclusion,
            reasoning_steps=reasoning_steps,
            confidence=final_confidence,
            alternative_interpretations=alternatives,
            requires_validation=requires_validation
        )
    
    def resolve_conflicts(self, conflicting_attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve conflicts between multiple attribute hypotheses.
        
        Args:
            conflicting_attributes: List of attribute hypotheses that conflict
            
        Returns:
            Resolved list of attributes with conflicts addressed
        """
        provider = self.llm_factory.get_available_provider()
        if not provider:
            return conflicting_attributes
        
        conflict_description = self._describe_conflicts(conflicting_attributes)
        
        resolution_prompt = f"""
        I have conflicting SEGY attribute hypotheses that need resolution:
        
        {conflict_description}
        
        Please analyze these conflicts and provide a resolution strategy. Consider:
        1. Which attributes are most likely correct based on SEGY standards
        2. Whether byte ranges can be adjusted to avoid overlaps
        3. If some hypotheses should be rejected entirely
        4. Priority rules for when conflicts cannot be resolved
        
        Provide your reasoning step by step and suggest the final attribute mappings.
        """
        
        response = provider.invoke_prompt(
            "You are resolving conflicts in SEGY header attribute parsing.",
            resolution_prompt,
            max_tokens=self.max_tokens_per_step
        )
        
        if response.success:
            # Parse the LLM response to extract resolved attributes
            # This is a simplified implementation - in practice, you'd want more sophisticated parsing
            return self._parse_resolution_response(response.content, conflicting_attributes)
        
        return conflicting_attributes
    
    def generate_alternatives(self, failed_hypothesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate alternative hypotheses when validation fails.
        
        Args:
            failed_hypothesis: The hypothesis that failed validation
            
        Returns:
            List of alternative hypotheses to try
        """
        provider = self.llm_factory.get_available_provider()
        if not provider:
            return []
        
        alternatives_prompt = f"""
        A SEGY attribute hypothesis has failed validation:
        
        Attribute: {failed_hypothesis.get('attribute_name', 'Unknown')}
        Byte Range: {failed_hypothesis.get('byte_start', 0)}-{failed_hypothesis.get('byte_end', 0)}
        Data Type: {failed_hypothesis.get('data_type', 'Unknown')}
        Failure Reason: {failed_hypothesis.get('validation_error', 'Unknown')}
        
        Please suggest alternative hypotheses to try. Consider:
        1. Different byte ranges (nearby locations, alignment issues)
        2. Alternative data types (int16, int32, float32, float64)
        3. Endianness considerations
        4. Standard SEGY locations for this attribute type
        
        Provide 3-5 specific alternative hypotheses with reasoning.
        """
        
        response = provider.invoke_prompt(
            "You are generating alternative SEGY parsing hypotheses after validation failure.",
            alternatives_prompt,
            max_tokens=self.max_tokens_per_step
        )
        
        if response.success:
            return self._parse_alternatives_response(response.content, failed_hypothesis)
        
        return []
    
    def _create_analysis_prompt(self, context: str, hypotheses: List[str]) -> str:
        """Create prompt for initial content analysis."""
        hypotheses_text = "\n".join([f"- {h}" for h in hypotheses])
        return f"""
        Analyze this SEGY textual header content for attribute-to-byte mappings:
        
        HEADER CONTENT:
        {context}
        
        POSSIBLE INTERPRETATIONS:
        {hypotheses_text}
        
        Step 1: What specific information can you extract from this header content?
        Look for:
        - Explicit byte location mentions (e.g., "bytes 73-76")
        - Attribute names and their descriptions
        - Data format indicators
        - Any numerical patterns or ranges
        
        Provide detailed analysis of what you can determine with certainty vs. what requires inference.
        """
    
    def _create_hypothesis_evaluation_prompt(self, context: str, hypotheses: List[str]) -> str:
        """Create prompt for hypothesis evaluation."""
        hypotheses_text = "\n".join([f"{i+1}. {h}" for i, h in enumerate(hypotheses)])
        return f"""
        Based on the previous analysis, evaluate each hypothesis:
        
        CONTEXT:
        {context}
        
        HYPOTHESES TO EVALUATE:
        {hypotheses_text}
        
        Step 2: For each hypothesis, assess:
        - How well does it match the textual evidence?
        - Is it consistent with SEGY format conventions?
        - What is the confidence level (high/medium/low)?
        - What are the potential risks or uncertainties?
        
        Rank the hypotheses from most to least likely.
        """
    
    def _create_standards_prompt(self, context: str) -> str:
        """Create prompt for SEGY standards consideration."""
        return f"""
        Apply SEGY format standards to validate the analysis:
        
        CURRENT ANALYSIS:
        {context}
        
        Step 3: Consider SEGY standards:
        - What are the standard byte locations for these attribute types?
        - Are the proposed locations consistent with SEGY revision formats?
        - Do the byte ranges align properly (2, 4, 8-byte boundaries)?
        - Are there any format violations or unusual patterns?
        
        How do standards support or contradict the current hypotheses?
        """
    
    def _create_synthesis_prompt(self, context: str, steps: List[ReasoningStep]) -> str:
        """Create prompt for final synthesis."""
        steps_summary = "\n".join([f"Step {s.step_number}: {s.conclusion}" for s in steps])
        return f"""
        Synthesize the chain-of-thought analysis into a final decision:
        
        FULL CONTEXT:
        {context}
        
        REASONING STEPS:
        {steps_summary}
        
        Final Step: Provide your conclusion including:
        1. The most likely attribute-to-byte mappings
        2. Confidence level for each mapping (0.0-1.0)
        3. Alternative interpretations to consider
        4. Whether data validation is required to confirm
        5. Any remaining uncertainties or risks
        
        Format your response clearly with specific byte ranges and confidence scores.
        """
    
    def _extract_conclusion(self, response_content: str) -> str:
        """Extract the main conclusion from LLM response."""
        # Simple extraction - look for conclusion indicators
        lines = response_content.split('\n')
        for line in lines:
            if any(indicator in line.lower() for indicator in ['conclusion:', 'result:', 'decision:', 'final:']):
                return line.split(':', 1)[-1].strip()
        
        # If no explicit conclusion, return first substantial line
        for line in lines:
            if len(line.strip()) > 20:
                return line.strip()
        
        return response_content[:200] + "..." if len(response_content) > 200 else response_content
    
    def _estimate_confidence(self, response_content: str) -> float:
        """Estimate confidence from LLM response content."""
        content_lower = response_content.lower()
        
        # Look for explicit confidence indicators
        if 'high confidence' in content_lower or 'very confident' in content_lower:
            return 0.9
        elif 'medium confidence' in content_lower or 'moderately confident' in content_lower:
            return 0.7
        elif 'low confidence' in content_lower or 'uncertain' in content_lower:
            return 0.4
        elif 'very uncertain' in content_lower or 'unclear' in content_lower:
            return 0.2
        
        # Look for uncertainty indicators
        uncertainty_words = ['maybe', 'possibly', 'might', 'could', 'uncertain', 'unclear', 'ambiguous']
        certainty_words = ['definitely', 'clearly', 'obviously', 'certain', 'confident', 'sure']
        
        uncertainty_count = sum(1 for word in uncertainty_words if word in content_lower)
        certainty_count = sum(1 for word in certainty_words if word in content_lower)
        
        if certainty_count > uncertainty_count:
            return 0.8
        elif uncertainty_count > certainty_count:
            return 0.5
        else:
            return 0.6  # Default moderate confidence
    
    def _extract_alternatives(self, response_content: str) -> List[str]:
        """Extract alternative interpretations from response."""
        alternatives = []
        lines = response_content.split('\n')
        
        in_alternatives_section = False
        for line in lines:
            line = line.strip()
            if 'alternative' in line.lower() and ('interpretation' in line.lower() or 'option' in line.lower()):
                in_alternatives_section = True
                continue
            
            if in_alternatives_section and line:
                if line.startswith('-') or line.startswith('•') or line[0].isdigit():
                    alternatives.append(line.lstrip('-•0123456789. '))
                elif len(alternatives) > 0:  # End of alternatives section
                    break
        
        return alternatives[:3]  # Limit to top 3 alternatives
    
    def _requires_validation(self, response_content: str) -> bool:
        """Determine if data validation is required."""
        content_lower = response_content.lower()
        validation_indicators = [
            'validation required', 'needs validation', 'should validate',
            'verify with data', 'check against trace', 'uncertain',
            'requires confirmation', 'needs verification'
        ]
        
        return any(indicator in content_lower for indicator in validation_indicators)
    
    def _describe_conflicts(self, conflicting_attributes: List[Dict[str, Any]]) -> str:
        """Create a description of attribute conflicts."""
        description = "Conflicting attribute hypotheses:\n\n"
        
        for i, attr in enumerate(conflicting_attributes, 1):
            description += f"{i}. {attr.get('attribute_name', 'Unknown')}\n"
            description += f"   Bytes: {attr.get('byte_start', 0)}-{attr.get('byte_end', 0)}\n"
            description += f"   Type: {attr.get('data_type', 'Unknown')}\n"
            description += f"   Confidence: {attr.get('confidence', 0.0):.2f}\n"
            description += f"   Source: {attr.get('source', 'Unknown')}\n\n"
        
        return description
    
    def _parse_resolution_response(self, response_content: str, original_attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse LLM response to extract resolved attributes."""
        # Simplified implementation - return original attributes with adjusted confidence
        # In a full implementation, you'd parse the LLM response more thoroughly
        resolved = []
        for attr in original_attributes:
            # Lower confidence for conflicted attributes
            attr_copy = attr.copy()
            attr_copy['confidence'] = max(0.1, attr_copy.get('confidence', 0.5) * 0.8)
            resolved.append(attr_copy)
        
        return resolved
    
    def _parse_alternatives_response(self, response_content: str, failed_hypothesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse LLM response to extract alternative hypotheses."""
        # Simplified implementation - generate some basic alternatives
        alternatives = []
        original_start = failed_hypothesis.get('byte_start', 0)
        original_end = failed_hypothesis.get('byte_end', 0)
        
        # Try nearby byte locations
        for offset in [-4, -2, 2, 4]:
            alt = failed_hypothesis.copy()
            alt['byte_start'] = max(0, original_start + offset)
            alt['byte_end'] = original_end + offset
            alt['confidence'] = 0.3
            alt['source'] = 'alternative_generation'
            alternatives.append(alt)
        
        # Try different data types
        data_types = ['int16', 'int32', 'float32', 'float64']
        current_type = failed_hypothesis.get('data_type', 'int32')
        for dtype in data_types:
            if dtype != current_type:
                alt = failed_hypothesis.copy()
                alt['data_type'] = dtype
                alt['confidence'] = 0.4
                alt['source'] = 'alternative_generation'
                alternatives.append(alt)
                break  # Just try one alternative type
        
        return alternatives[:3]  # Limit to 3 alternatives