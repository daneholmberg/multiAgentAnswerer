# Parsing and Error Handling Improvements

This document outlines the areas in the codebase that need improved parsing and error handling for production use.

## Current Simplified Implementations

The utils.py file contains several simplified implementations with comments indicating they need improvements:

### 1. Evaluation Parsing

```python
def parse_evaluation_result(evaluation_text: str, question: str, evaluator_id: str) -> Evaluation:
    """
    Parse the evaluation text from an agent into structured evaluation data.
    
    This is complex and might require some prompt engineering to get the agent
    to return data in a specific format.
    """
    # This is a simplified version - in a real implementation, 
    # you might need more robust parsing
    try:
        # Try parsing as JSON first
        if "```json" in evaluation_text:
            json_start = evaluation_text.find("```json") + 7
            json_end = evaluation_text.find("```", json_start)
            data = json.loads(evaluation_text[json_start:json_end].strip())
        else:
            data = json.loads(evaluation_text)
        
        # ... rest of the function ...
    
    except Exception as e:
        logger.error(f"Error parsing evaluation result: {e}")
        logger.debug(f"Raw evaluation text: {evaluation_text}")
        
        # Fallback - return an empty evaluation
        return Evaluation(
            criteria=[],
            scores=[],
            evaluator_id=evaluator_id,
            question=question
        )
```

### 2. Improvement Parsing

```python
def parse_improvement_result(improvement_text: str, original_answer_id: str, agent_id: str) -> ImprovedAnswer:
    """
    Parse the improvement text from an agent into an ImprovedAnswer object.
    """
    # In a real implementation, you might need more robust parsing
    try:
        if "```json" in improvement_text:
            json_start = improvement_text.find("```json") + 7
            json_end = improvement_text.find("```", json_start)
            data = json.loads(improvement_text[json_start:json_end].strip())
            
            # ... rest of the function ...
            
        else:
            # Simpler fallback - assume the entire text is the improved answer
            return ImprovedAnswer(
                original_answer_id=original_answer_id,
                content=improvement_text,
                agent_id=agent_id,
                improvements=""
            )
    except Exception as e:
        logger.error(f"Error parsing improvement result: {e}")
        logger.debug(f"Raw improvement text: {improvement_text}")
        
        # Fallback - return the original text
        return ImprovedAnswer(
            original_answer_id=original_answer_id,
            content=improvement_text,
            agent_id=agent_id,
            improvements=""
        )
```

## Required Improvements

### 1. Robust JSON Parsing

- **Issue**: Current parsing is basic and doesn't handle many edge cases
- **Improvements Needed**:
  - Add support for multiple JSON formats ("```json", "```", plain text)
  - Handle malformed JSON more gracefully
  - Add validation for expected fields
  - Implement schema validation using Pydantic models
  - Add better logging of parse failures

### 2. Error Recovery Strategies

- **Issue**: Basic fallbacks with minimal information
- **Improvements Needed**:
  - Implement more sophisticated fallback strategies
  - Add partial data recovery when possible
  - Provide more informative error messages
  - Consider retry logic for certain failures
  - Track and report parsing success rates

### 3. Structured Response Templates

- **Issue**: LLMs may not consistently format responses as requested
- **Improvements Needed**:
  - Enhance prompt engineering to increase consistency
  - Provide clearer response templates in task descriptions
  - Consider implementing a parser that can handle semi-structured text
  - Add examples in prompts to guide response formatting

### 4. Custom Parsing Methods

- **Issue**: Generic parsing for all LLMs
- **Improvements Needed**:
  - Create model-specific parsing methods
  - Tune parsing based on known response patterns of each LLM
  - Validate results more thoroughly for each model
  - Add adaptive parsing that learns from successful patterns

### 5. API Error Handling

- **Issue**: Basic exception handling for API calls
- **Improvements Needed**:
  - Add specific handling for common API errors (rate limits, authentication issues)
  - Implement retry logic with exponential backoff
  - Add circuit breakers for failing APIs
  - Provide fallback options when an API is unavailable

## Implementation Priority

1. **High Priority**:
   - Enhance JSON parsing with better format detection
   - Add validation for expected fields
   - Implement proper error logging

2. **Medium Priority**:
   - Create model-specific parsing methods
   - Add retry logic for API calls
   - Improve prompt engineering for consistent responses

3. **Low Priority**:
   - Implement schema validation
   - Add adaptive parsing
   - Create comprehensive parsing metrics

## Testing Strategy

For each parsing improvement:

1. Create test cases with various LLM response formats
2. Test with intentionally malformed responses
3. Verify recovery mechanisms work as expected
4. Measure parsing success rates before and after improvements 