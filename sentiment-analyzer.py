import json
import pandas as pd
import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole

class ConversationSummarizer:
    def __init__(self, api_key=None):
        """
        Initialize the ConversationSummarizer with LlamaIndex OpenAI
        
        Args:
            api_key: Optional API key. If not provided, will use environment variable
        """
        self.llm = OpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.15
        )
        self.schema = {
            "type": "object",
            "properties": {
                "conversation_metadata": {
                    "type": "object",
                    "properties": {
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "call_type": {
                            "type": "string",
                            "enum": ["inbound", "outbound"]
                        },
                        "customer_name": {
                            "type": "string"
                        },
                        "account_type": {
                            "type": "string",
                            "enum": [
                                "Checking accounts",
                                "Savings accounts",
                                "Money market accounts (MMAs)",
                                "Certificate of deposit (CD) accounts"
                            ]
                        },
                        "agent_name": {
                            "type": "string"
                        },
                        "account_number": {
                            "type": "string"
                        }
                    },
                    "required": ["timestamp", "call_type", "customer_name", "account_type"]
                },
                "conversation_details": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "object",
                            "properties": {
                                "overview": {
                                    "type": "string"
                                },
                                "primary_issue": {
                                    "type": "string"
                                },
                                "resolution_status": {
                                    "type": "string",
                                    "enum": ["resolved", "pending", "escalated"]
                                },
                                "call_outcome": {
                                    "type": "string",
                                    "enum": ["resolved", "escalated", "callback_scheduled", "unresolved"]
                                }
                            },
                            "required": ["overview", "primary_issue", "resolution_status", "call_outcome"]
                        },
                        "transcript_segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "timestamp": {
                                        "type": "string",
                                        "format": "date-time"
                                    },
                                    "speaker": {
                                        "type": "string",
                                        "enum": ["customer", "agent"]
                                    },
                                    "text": {
                                        "type": "string"
                                    },
                                    "sentiment": {
                                        "type": "string",
                                        "enum": ["positive", "neutral", "negative"]
                                    },
                                    "key_phrases": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                },
                                "required": ["timestamp", "sentiment", "speaker", "text", "key_phrases"]
                            }
                        }
                    },
                    "required": ["summary", "transcript_segments"]
                },
                "key_points": {
                    "type": "object",
                    "properties": {
                        "customer_concerns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["account_issue", "fraud", "transaction_dispute", 
                                                "loan_inquiry", "credit_card_issue", "general_inquiry"]
                                    },
                                    "description": {
                                        "type": "string"
                                    },
                                    "transaction_reference": {
                                        "type": "object",
                                        "properties": {
                                            "amount": {
                                                "type": "number"
                                            },
                                            "date": {
                                                "type": "string",
                                                "format": "date"
                                            },
                                            "account_last4": {
                                                "type": "string"
                                            }
                                        }
                                    }
                                },
                                "required": ["type", "description"]
                            }
                        }
                    }
                },
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["follow_up_call", "document_verification", "internal_escalation"]
                            },
                            "action_type": {
                                "type": "string",
                                "enum": ["follow_up", "issue_resolution", "document_submission", 
                                        "callback", "refund_process"]
                            },
                            "assigned_to": {
                                "type": "string"
                            },
                            "due_date": {
                                "type": "string",
                                "format": "date-time"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["open", "in_progress", "completed"]
                            }
                        },
                        "required": ["type", "action_type", "assigned_to", "due_date", "status"]
                    }
                },
                "analytics": {
                    "type": "object",
                    "properties": {
                        "overall_sentiment": {
                            "type": "string",
                            "enum": ["positive", "neutral", "negative"]
                        }
                    }
                }
            }
        }
    
    def process_conversation(self, conversation_text):
        """
        Process conversation text and extract structured information
        
        Args:
            conversation_text: The conversation transcript
            
        Returns:
            Dictionary containing structured conversation analysis
        """
        prompt = f"""
Analyze this conversation and generate a structured schema as per the following steps:

1. Provide Metadata:
   - Participants (customer and agent)
   - Duration (in seconds)

2. Generate a Summary:
   - Main issue (1-2 words)
   - Customer concerns
   - Resolution status: Understand the whole conversation and based on the end status, classify one among [resolved, unresolved, pending]

3. Perform Sentiment Analysis:
   - Determine the overall sentiment trend of the entire conversation by analyzing the tone, language, and context
   - Classify the overall sentiment as:
     - 'positive': customer exhibits satisfaction, gratitude, or constructive engagement
     - 'neutral': conversation reflects a balanced tone with no strong emotions
     - 'negative': customer exhibits frustration, dissatisfaction, or complaints
   - Populate: "analytics": one among positive, neutral, negative"
   - Give only one sentiment based on whole conversation

4. Extract Key Phrases (1-2 Words):
   - Identify concise key transaction details (e.g., "contribution error", "fraudulent charge")
   - Pinpoint issues or topics related to financial domain(e.g., "account error", "policy confusion")

5. Identify Customer Concerns (1-2 Words):
   - Summarize customer concerns succinctly (e.g., "fraud", "billing", "confusion")

6. Track Agent Actions (1-2 Words):
   - type should have one among "follow_up_call", "document_verification", "internal_escalation"
   - assigned_to should have either the name of the employee (the agent) or the relevant department
   - due_date should contain the date
   - status should have one among "open", "in_progress", "completed"

7. Analytics:
   - The overall sentiment of the conversation

Key Notes:
- Ensure that all extracted content is limited to 1-2 words while still conveying the main meaning
- The overall sentiment trend should reflect the tone of the entire conversation
- Do not leave any JSON fields of the schema blank
- Return ONLY valid JSON matching the schema

Conversation:
{conversation_text}

Return the analysis as a JSON object following this schema:
{json.dumps(self.schema, indent=2)}
"""
        
        try:
            # Create chat message
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant that analyzes customer service conversations and returns structured JSON data."),
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            
            # Get response from LLM
            response = self.llm.chat(messages)
            
            # Parse the response
            response_text = response.message.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Parse JSON
            summary = json.loads(response_text.strip())
            
            return summary
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response: {str(e)}")
            st.error(f"Response text: {response_text[:500]}...")
            return None
        except Exception as e:
            st.error(f"Error processing conversation: {str(e)}")
            return None
    
    def tag_dialogue_with_llm(self, transcript):
        prompt = f"""Given the following conversation transcript between a customer and a bank agent, classify each line with the speaker.
            Rules for classification:
            1. Agents typically:
            - Introduce themselves and their company
            - Ask for verification information
            - Use formal language and company protocols
            - Provide solutions and explanations
            - End conversations with closing statements

            2. Customers typically:
            - Describe problems or make requests
            - Provide personal information when asked
            - Ask questions about services/issues
            - Respond to agent's questions
            - Express satisfaction/dissatisfaction

            Format each line as either 'Customer: <text>' or 'Agent: <text>'.

            Transcript:
            {transcript}
            """
        
        try:
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM, 
                    content="You are a helpful assistant that labels conversation roles in customer service transcripts."
                ),
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            
            response = self.llm.chat(messages)
            return response.message.content
            
        except Exception as e:
            st.error(f"Error tagging dialogue: {str(e)}")
            return transcript  # Return original transcript if tagging fails


def style_sentiment_dataframe(df):
    """Add emoji indicators to sentiment columns in the DataFrame"""
    styled_df = df.copy()
    
    # Identify columns that might contain sentiment values
    sentiment_columns = [
        col for col in df.columns 
        if any(term in col.lower() for term in ['sentiment', 'emotion', 'feeling', 'tone'])
    ]
    
    # Apply emoji mapping to sentiment columns
    for col in sentiment_columns:
        styled_df[col] = df[col].apply(
            lambda x: f"🔴 {x}" if 'negative' in str(x).lower() 
            else f"🟢 {x}" if 'positive' in str(x).lower() 
            else f"⚪ {x}"
        )
    
    return styled_df


def summary_to_dataframe(summary):
    """
    Convert summary dictionary to pandas DataFrames for display
    
    Args:
        summary: Dictionary containing conversation analysis
        
    Returns:
        Dictionary of DataFrames for different aspects of the analysis
    """
    # Display overall sentiment if available
    analytics_df = pd.json_normalize(summary.get("analytics", {}))
    
    if not analytics_df.empty and "overall_sentiment" in analytics_df.columns:
        overall_sentiment = analytics_df["overall_sentiment"].iloc[0]
        sentiment_indicator = (
            "🔴" if 'negative' in str(overall_sentiment).lower() 
            else "🟢" if 'positive' in str(overall_sentiment).lower() 
            else "⚪"
        )
        
        st.markdown(f"""
        <div style='text-align: left; padding: 20px;'>
            <h3>Overall Sentiment: {sentiment_indicator} {overall_sentiment}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Return all dataframes excluding Analytics
    dataframes = {
        "Metadata": pd.json_normalize(summary.get("conversation_metadata", {})),
        "Summary": pd.json_normalize(summary.get("conversation_details", {}).get("summary", {})),
        "Detailed Transcript": pd.json_normalize(
            summary.get("conversation_details", {}).get("transcript_segments", [])
        ),
        "Customer Concerns": pd.json_normalize(
            summary.get("key_points", {}).get("customer_concerns", [])
        ),
        "Action Items": pd.json_normalize(summary.get("action_items", []))
    }
    
    return dataframes


def get_sentiment_summary_stats(summary):
    """
    Extract key statistics from sentiment analysis
    
    Args:
        summary: Dictionary containing conversation analysis
        
    Returns:
        Dictionary of summary statistics
    """
    stats = {}
    
    # Overall sentiment
    if "analytics" in summary:
        stats["overall_sentiment"] = summary["analytics"].get("overall_sentiment", "N/A")
    
    # Resolution status
    if "conversation_details" in summary and "summary" in summary["conversation_details"]:
        conv_summary = summary["conversation_details"]["summary"]
        stats["resolution_status"] = conv_summary.get("resolution_status", "N/A")
        stats["call_outcome"] = conv_summary.get("call_outcome", "N/A")
        stats["primary_issue"] = conv_summary.get("primary_issue", "N/A")
    
    # Count of customer concerns
    if "key_points" in summary and "customer_concerns" in summary["key_points"]:
        stats["concern_count"] = len(summary["key_points"]["customer_concerns"])
    
    # Count of action items
    if "action_items" in summary:
        stats["action_item_count"] = len(summary["action_items"])
    
    return stats
