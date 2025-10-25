# Updated with two columns side by side

import streamlit as st
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from typing import List, Dict, Any
import os
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts import PromptTemplate
from llama_index.core.readers.json import JSONReader
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import (
    AgentInput, AgentOutput, ToolCall, ToolCallResult, AgentStream,
)
import asyncio
import logging
import json
import os
from typing import Dict, Any
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from utils import *
from timeline_function import create_timeline


from sentiment_analyzer import (
    ConversationSummarizer, 
    summary_to_dataframe, 
    style_sentiment_dataframe,
    get_sentiment_summary_stats
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Streamlit page config
st.set_page_config(
    page_title="Customer Service Analysis Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agent_context' not in st.session_state:
    st.session_state.agent_context = AgentContext()
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# Initialize LLM and settings
@st.cache_resource
def initialize_llm():
    try:
        llm = OpenAI(model="gpt-4o-mini")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# Load data and create indexes
@st.cache_resource
def load_data_indexes():
    try:
        collated_json = SimpleDirectoryReader(
            input_files=["paysafe/conversation_data.json"]
        ).load_data()
        transaction_doc = SimpleDirectoryReader(
            input_files=["paysafe/transaction data.json"]
        ).load_data()
        
        collated_index = VectorStoreIndex.from_documents(collated_json)
        transaction_index = VectorStoreIndex.from_documents(transaction_doc)
        
        collated_engine = collated_index.as_query_engine(similarity_top_k=3)
        transaction_engine = transaction_index.as_query_engine(similarity_top_k=3)
        
        return collated_engine, transaction_engine
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Utility functions
def load_transcript_files(directory: str) -> Dict[str, Dict[str, str]]:
    """Load all transcript files from the directory"""
    try:
        if not os.path.exists(directory):
            st.error(f"Directory not found: {directory}")
            return {}
        
        transcript_files = {}
        for file_path in Path(directory).rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                transcript_files[file_path.name] = {
                    'path': str(file_path),
                    'content': content,
                    'relative_path': str(file_path.relative_to(directory))
                }
            except Exception as e:
                st.warning(f"Could not read file {file_path.name}: {str(e)}")
        return transcript_files
    except Exception as e:
        st.error(f"Error loading transcript files: {str(e)}")
        return {}

def get_folder_structure(directory: str) -> Dict:
    """Get the folder structure for navigation"""
    try:
        if not os.path.exists(directory):
            return {}
        
        structure = {}
        for root, dirs, files in os.walk(directory):
            txt_files = [f for f in files if f.endswith('.txt')]
            if txt_files:
                rel_path = os.path.relpath(root, directory)
                if rel_path == '.':
                    rel_path = 'Root'
                structure[rel_path] = {
                    'full_path': root,
                    'files': txt_files
                }
        return structure
    except Exception as e:
        st.error(f"Error getting folder structure: {str(e)}")
        return {}

def process_transcript_content(transcript_content: str) -> str:
    """Process transcript content directly instead of reading from a file."""
    try:
        if not transcript_content:
            return json.dumps({"error": "Transcript content is empty"})
        
        result = json.dumps({
            "content": transcript_content,
            "source": "direct_input"
        }, ensure_ascii=False)
        
        return result
    except Exception as e:
        return json.dumps({"error": f"Error processing transcript: {str(e)}"})

def parse_rca_response_simple(response: str) -> dict:
    """Simple and effective RCA response parser."""
    analysis_results = {}
    current_key = None
    current_content = []
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('ROOT_CAUSE:'):
            if current_key and current_content:
                analysis_results[current_key] = ' '.join(current_content)
            current_key = 'rootcause'
            current_content = [line.split('ROOT_CAUSE:', 1)[1].strip()]
        elif line.startswith('PATTERNS:'):
            if current_key and current_content:
                analysis_results[current_key] = ' '.join(current_content)
            current_key = 'pattern'
            current_content = [line.split('PATTERNS:', 1)[1].strip()]

        elif line.startswith('RATIONALE:'):
            if current_key and current_content:
                analysis_results[current_key] = ' '.join(current_content)
            current_key = 'rationale'
            current_content = [line.split('RATIONALE:', 1)[1].strip()]
        elif current_key:
            current_content.append(line)
    
    # Don't forget the last section
    if current_key and current_content:
        analysis_results[current_key] = ' '.join(current_content)
    
    return analysis_results

# Tool functions (simplified for Streamlit)
def create_tools(llm, collated_engine, transaction_engine, agent_context):
    """Create all the tools for the agent"""
    
    def decide_context_tool_fn() -> str:
        decision_prompt = f"""
        Based on the issue and transcript below, decide whether conversation context is enough or if both conversation and transaction data are needed.
        Issue: {agent_context.get('issue')}
        Transcript: {agent_context.get('summary')}
        Respond with: 'conversation_only' or 'both'
        """
        decision = llm.complete(decision_prompt)
        agent_context.update({"context_decision": decision.text.strip()})
        return decision.text.strip()

    def run_conversation_tool() -> str:
        name = agent_context.get("name")
        issue = agent_context.get("issue")
        query = f"Customer: {name}. Issue: {issue}. Provide detailed conversation context."
        response = collated_engine.query(query)
        # print(response)
        context_text = "\n\n".join([node.node.get_content() for node in response.source_nodes])
        # print(context_text)
        agent_context.update({"conversation_context": context_text})
        return "Conversation context retrieved and stored."

    def run_transaction_tool() -> str:
        name = agent_context.get("name")
        issue = agent_context.get("issue")
        query = f"Customer: {name}. Issue: {issue}. Provide detailed transaction context."
        response = transaction_engine.query(query)
        context_text = "\n\n".join([node.node.get_content() for node in response.source_nodes])
        agent_context.update({"transaction_context": context_text})
        return "Transaction context retrieved and stored."

    def analyze_content() -> Dict[str, Any]:
        """Analyze transcript content for summary, issue, and status using global context."""
        transcript = agent_context.get('transcript')
        if not transcript:
            return {"error": "No transcript content available in context"}

        content_json = process_transcript_content(transcript)
        if not content_json:
            return {"error": "Empty input received"}

        try:
            content_data = json.loads(content_json)
        except json.JSONDecodeError:
            content_data = {"content": content_json}

        content = content_data.get("content", "")
        if not content:
            return {"error": "No content found in input"}

        # Summary analysis
        summary_prompt = PromptTemplate(
            "Summarize the following customer service transcript in 2-3 sentences, focusing on the main issue and resolution:\n\n{content}\n\nSummary:"
        )
        summary = llm.complete(summary_prompt.format(content=content))

        # Name extraction
        name_prompt = PromptTemplate(
            "Extract the customer's name from this transcript (not the agent). Return only the name:\n\n{content}\n\nCustomer Name:"
        )
        name = llm.complete(name_prompt.format(content=content))

        # Issue identification
        issue_prompt = PromptTemplate(
            "From the following transcript, identify the main issue in exactly 2-3 words:\n\n{content}\n\nMain Issue:"
        )
        issue = llm.complete(issue_prompt.format(content=content))

        # Status determination
        status_prompt = PromptTemplate(
            "From the following transcript, determine if the issue was 'resolved' or 'pending'. Return only one word:\n\n{content}\n\nStatus:"
        )
        status = llm.complete(status_prompt.format(content=content))

        # Summary analysis
        overall_sentiment = PromptTemplate(
            "Identify the overall sentiment of the conversation one among the following [[Positive], [Neutral], [Negative]]. Return only one word:\n\n{content}\n\nSentiment:"
        )
        sentiment = llm.complete(overall_sentiment.format(content=content))

        extracted_data = {
            "summary": str(summary.text).strip(),
            "name": str(name.text).strip(),
            "issue": str(issue.text).strip(),
            "status": str(status.text).strip(),
            "sentiment": str(sentiment.text).strip()
        }

        agent_context.update(extracted_data)

        return {
            "message": "Content analysis completed successfully",
            "extracted_data": extracted_data,
            "context_status": "updated"
        }

    def perform_root_cause_analysis() -> dict:
        prompt = f"""
        Perform root cause analysis for the issue.
        Name: {agent_context.get('name')}
        Issue: {agent_context.get('issue')}
        Conversation Context: {agent_context.get('conversation_context') if agent_context.get('context_decision') == 'conversation_only' else ""}
        {"Transaction Context: " + str(agent_context.get('transaction_context')) if agent_context.get('context_decision') == 'both' else ""}
        The root cause should match with the extracted names and key concern. Do not hallucinate.

        1. Timeline Integration Analysis
        Extract all timestamps from both the knowledge base context and transaction data
        Create a unified chronological timeline incorporating both data sets
        Identify temporal correlations between customer interactions/issues and financial transactions
        Examine transaction patterns (withdrawals, deposits, liquidations) in relation to key events
        Complete the timeline in 6 exact sentences include important events

        2. Root Cause Identification
        Explain everything in 5 W's: why is the customer dissatisfied, what is the issue, what steps are taken, what is the result

        3. Pattern Recognition
        Identify recurring patterns or trends across the data
        Note specific triggers that appear to initiate transaction behaviors
        Identify common failure points where customer issues and transaction problems intersect
        Detect unusual transaction patterns and their relation to customer interactions

        REQUIRED OUTPUT FORMAT
        Your analysis must be provided in exactly the following format with these precise section headers:

        ROOT_CAUSE: Explain everything in 5 W's: why is the customer dissatisfied, when did this happen, what is the issue, what steps are taken, what is the result?

        PATTERNS: Detail the recurring patterns discovered across both datasets. Include specific triggers, common failure points, and any unusual transaction behaviors (continuous withdrawals/deposits, liquidation patterns, etc.) that correlate with customer interactions. Highlight significant transaction anomalies and their connection to customer issues. Explain all these in 5 sentences short and concise.

        RATIONALE: Explain your reasoning process and why you believe these connections exist between the knowledge base context and transaction data. Include your analysis of how customer behavior correlates with transaction activities and what process failures might be occurring. Provide evidence-based justification for your conclusions and show me the numbers from the transaction data that is related to the analysis. Also provide few actionable recommendations.
        """

        response = llm.complete(prompt)
        data = {"rootcause": str(response.text)}
        agent_context.update(data)

        return {
            "message": "Root cause analysis completed successfully",
            "context_status": "updated"
        }
    
    def timeline_analysis() -> dict:
        prompt = f"""
        Perform root cause analysis for the issue.
        Name: {agent_context.get('name')}
        Issue: {agent_context.get('issue')}
        Conversation Context: {agent_context.get('conversation_context') if agent_context.get('context_decision') == 'conversation_only' else ""}
        {"Transaction Context: " + str(agent_context.get('transaction_context')) if agent_context.get('context_decision') == 'both' else ""}

        Perform a timeline Analysis
        1. Timeline Integration Analysis
        - Extract all timestamps from both the knowledge base context and transaction data.
        - Create a unified chronological timeline incorporating both datasets.
        - Identify temporal correlations between customer interactions/issues and financial transactions.
        - Examine transaction patterns (withdrawals, deposits, liquidations) in relation to key events.
        - Complete the timeline including important events.

        ### REQUIRED OUTPUT FORMAT
        TIMELINE:
        Present a chronological organization of all relevant events from both datasets.
        Include specific dates from both contexts, showing how they correlate.
        Highlight important milestones, decision points, initial contacts, escalations, follow-up actions, and resolution attempts.
        Show clear connections between dates in the knowledge base and transaction activities.

        Example format:
        October 1st, 2023: Daniel Patel initiated a wire transfer.
        October 3rd, 2023: Daniel Patel contacted customer support regarding the delay.
        ...
        Ensure exactly 6 sentences are produced, each representing a chronological event.
        """

        response = llm.complete(prompt)
        data = {"timeline": str(response.text)}
        agent_context.update(data)

        return {
            "message": "Timeline Generated",
            "context_status": "updated"
        }

    def check_context_status() -> Dict[str, Any]:
        """Check the current context status and determine next action."""
        required_fields = ['name', 'issue', 'status']
        missing_fields = []

        for field in required_fields:
            try:
                value = agent_context.get(field)
                if not value or (isinstance(value, str) and value.strip() == ''):
                    missing_fields.append(field)
            except Exception:
                missing_fields.append(field)

        if missing_fields:
            return {
                'next_action': 'extract_data',
                'status': 'incomplete',
                'missing_fields': missing_fields,
                'message': f"Need to extract: {', '.join(missing_fields)}"
            }

        try:
            rootcause = agent_context.get('rootcause')
        except Exception:
            rootcause = None

        if not rootcause:
            try:
                status = agent_context.get('status')
                if not status:
                    status = ''
                status = str(status).lower()

                if any(keyword in status for keyword in ['pending', 'open', 'unresolved', 'escalated']):
                    return {
                        'next_action': 'perform_rca',
                        'status': 'ready_for_rca',
                        'message': 'All data extracted. Ready for root cause analysis.'
                    }
                elif any(keyword in status for keyword in ['resolved', 'closed', 'completed']):
                    return {
                        'next_action': 'perform_rca',
                        'status': 'resolved_but_rca_recommended',
                        'message': 'Issue resolved but RCA recommended for process improvement.'
                    }
                else:
                    return {
                        'next_action': 'perform_rca',
                        'status': 'status_unclear',
                        'message': 'Status unclear. Performing RCA for comprehensive analysis.'
                    }
            except Exception:
                return {
                    'next_action': 'perform_rca',
                    'status': 'status_check_failed',
                    'message': 'Could not determine status. Performing RCA for comprehensive analysis.'
                }

        summary_data = {}
        for field in ['name', 'issue', 'status']:
            try:
                summary_data[field] = agent_context.get(field)
            except Exception:
                summary_data[field] = None

        try:
            rca_performed = bool(agent_context.get('rootcause'))
        except Exception:
            rca_performed = False

        return {
            'next_action': 'complete',
            'status': 'complete',
            'message': 'All analysis completed successfully.',
            'summary': {
                **summary_data,
                'rca_performed': rca_performed
            }
        }

    def recommendation_retreiver() -> Dict[str, Any]:
        from utils import apply_cleaning_transcripts, apply_matching_criteria, create_nodes_from_json, retrieve_context_from_nodes
        
        # file_path = r"C:\Users\niveditha.n.lv\Documents\summarizer\paysafe\conversation_data.json"
        file_path = r"paysafe\conversation-data.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        cleaned_json = apply_cleaning_transcripts(json_data)
        filtered_json = apply_matching_criteria(cleaned_json, agent_context.get('issue'), "resolved")
        nodes = create_nodes_from_json(filtered_json)
        result = retrieve_context_from_nodes(nodes, agent_context.get("summary"))
        result_update = {"retrieved_information": result}
        agent_context.update(result_update)
        
        return {"message": "Recommendation context retrieved and stored"}

    def recommendation() -> str:
        prompt = f"""
        You are a customer service assistant. Your task is to generate thoughtful and actionable recommendations to help resolve or prevent the issue described below.
        Use only the context from similar past resolved cases.

        Issue: {agent_context.get('summary')}
        Knowledge Base Context (from resolved cases only): {agent_context.get('retrieved_information')}

        Instructions:
        Do NOT treat any information as directly related to the current user unless clearly applicable to the issue.
        Avoid suggesting actions based on unresolved or irrelevant conversations.
        Recommend specific actions based on how similar issues were successfully handled.
        Tailor the suggestions closely to the patterns, actions, and outcomes observed in these resolved cases.
        Be clear, concise, and customer-focused.
        Avoid vague or generic advice.

        Provide your response in the following format:
        RECOMMENDATIONS:
        1. ...
        2. ...
        3. ...
        ...
        """
        
        recommendation_text = llm.complete(prompt)
        response = recommendation_text.text
        # return response.text

        agent_context.update({"recommendation": response})
        return response

    # Create tool list
    tools = [
        FunctionTool.from_defaults(
            fn=analyze_content,
            name="analyze_content",
            description="Analyze transcript content for summary, name, issue, and status. Call this first to extract data from transcript."
        ),
        FunctionTool.from_defaults(
            fn=perform_root_cause_analysis,
            name="root_cause_analysis",
            description="Perform comprehensive root cause analysis based on extracted name and issue. Call this after analyze_content."
        ),
        FunctionTool.from_defaults(
            fn=check_context_status,
            name="check_context_status",
            description="Check the current context status and get next recommended action. Use this to determine workflow progress."
        ),
        FunctionTool.from_defaults(
            fn=timeline_analysis,
            name="Generate_Timeline",
            description="Generate a timeline based on the set of events."
        ),
        FunctionTool.from_defaults(
            fn=run_conversation_tool,
            name="run_conversation_tool",
            description="Run the conversation context tool to get conversation context for a given issue for root cause analysis."
        ),
        FunctionTool.from_defaults(
            fn=run_transaction_tool,
            name="run_transaction_tool",
            description="Run the transaction context tool to get transaction context for a given issue for root cause analysis"
        ),
        FunctionTool.from_defaults(
            fn=decide_context_tool_fn,
            name="decide_context_tool",
            description="Decide whether conversation context is enough or if both conversation and transaction data are needed based on the issue and transcript."
        ),
        FunctionTool.from_defaults(
            fn=recommendation_retreiver,
            name="recommendation_retreiver",
            description="Retrieves the important context needed for the recommendation tool based on the issue and context. Use this to provide specific, evidence-based suggestions for resolution or prevention."
        ),
        FunctionTool.from_defaults(
            fn=recommendation,
            name="recommendation",
            description="Generate actionable recommendations based on the issue and context. Use this to provide specific, evidence-based suggestions for resolution or prevention."
        )
    ]
    
    return tools






def generate_events(conversation_text):
    """
    Performs root cause analysis by generating a 6-sentence timeline
    correlating conversation events with knowledge base and transaction data.
    """

    prompt = f"""
        You are an analytical assistant performing root cause analysis for a customer issue.

        ### INPUT DATA
        Conversation:
        {conversation_text}

        ### TASK
        Perform root cause analysis for the issue.

        Timeline Integration Analysis
            - Extract all timestamps from both the knowledge base context and transaction data.
            - Create a unified chronological timeline incorporating both datasets.
            - Identify temporal correlations between customer interactions/issues and financial transactions.
            - Examine transaction patterns (withdrawals, deposits, liquidations) in relation to key events.
            - Complete the timeline including important events.

        Present a chronological organization of all relevant events from both datasets.
        Include specific dates from both contexts, showing how they correlate.
        Highlight important milestones, decision points, initial contacts, escalations, follow-up actions, and resolution attempts.
        Show clear connections between dates in the knowledge base and transaction activities.

        ### REQUIRED OUTPUT FORMAT

        October 1st, 2023: Daniel Patel initiated a wire transfer.
        October 3rd, 2023: Daniel Patel contacted customer support regarding the delay.
        ...
        Ensure exactly 6 sentences are produced, each representing a chronological event.
        """

    timeline_text = llm.complete(prompt)
    response = timeline_text.text
    print(response)

    return response




import matplotlib.pyplot as plt
import re
import textwrap

def parse_timeline_data(timeline_data):
    """Parse timeline entries to extract dates and descriptions"""
    parsed_data = []
    
    if isinstance(timeline_data, str):
        # Split by bullet points if it's already formatted
        if timeline_data.strip().startswith("•"):
            items = [item.strip()[2:].strip() for item in timeline_data.split("\n\n") if item.strip()]
        else:
            # Split by " - " if it's a string with separators
            items = [item.strip() for item in timeline_data.split(" - ") if item.strip()]
    elif isinstance(timeline_data, list):
        items = timeline_data
    else:
        return []
    
    for item in items:
        # Extract date using regex
        date_match = re.match(r'^([A-Za-z]+ \d+(?:st|nd|rd|th)?, \d{4}):', item)
        if date_match:
            date_str = date_match.group(1)
            description = item[len(date_match.group(0)):].strip()
            
            # Parse the date
            try:
                date_obj = datetime.strptime(date_str, "%B %dst, %Y")
            except ValueError:
                try:
                    date_obj = datetime.strptime(date_str, "%B %dnd, %Y")
                except ValueError:
                    try:
                        date_obj = datetime.strptime(date_str, "%B %drd, %Y")
                    except ValueError:
                        try:
                            date_obj = datetime.strptime(date_str, "%B %dth, %Y")
                        except ValueError:
                            # If all specific formats fail, try a more generic approach
                            cleaned_date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
                            date_obj = datetime.strptime(cleaned_date, "%B %d, %Y")
            
            parsed_data.append({
                'date': date_obj,
                'date_str': date_obj.strftime('%b %d, %Y'),
                'description': description
            })
    
    # Sort by date
    parsed_data.sort(key=lambda x: x['date'])
    return parsed_data

def create_matplotlib_vertical_timeline(timeline_data, figsize=(10, 8), save_path=None):
    """Create a vertical timeline visualization using Matplotlib"""
    parsed_data = parse_timeline_data(timeline_data)
    
    if not parsed_data:
        print("No valid timeline data to plot")
        return None
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate positions
    num_events = len(parsed_data)
    positions = list(range(num_events, 0, -1))  # Reversed for chronological order from top to bottom
    
    # Extract dates and descriptions
    dates = [item['date_str'] for item in parsed_data]
    descriptions = [item['description'] for item in parsed_data]
    
    # Draw vertical line
    ax.plot([0.1] * num_events, positions, 'o-', color='gray', markersize=12, markerfacecolor='#3498db')
    
    # Add dates on the left side
    for i, (date, pos) in enumerate(zip(dates, positions)):
        ax.text(0.05, pos, date, ha='right', va='center', fontsize=10)
    
    # Add descriptions on the right side
    for i, (desc, pos) in enumerate(zip(descriptions, positions)):
        # Wrap text for better display
        wrapped_text = textwrap.fill(desc, width=50)
        ax.text(0.15, pos, wrapped_text, ha='left', va='center', fontsize=9)
    
    # Set labels and title
    # ax.set_title('Event Timeline', fontsize=14, fontweight='bold')
    
    # Remove axes and ticks
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    
    # Set limits
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, num_events + 1)
    
    plt.tight_layout(pad=1.0, w_pad=0, h_pad=0, rect=[0.01, 0, 1, 0.95])
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig, ax

def apply_custom_css():
    """Apply custom CSS to use Manrope font throughout the app"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@200;300;400;500;600;700;800&display=swap');
    
    /* Apply Manrope to all text elements */
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }
    
    /* Specifically target Streamlit elements */
    .stApp {
        font-family: 'Manrope', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Manrope', sans-serif !important;
    }
    
    /* Text elements */
    p, div, span, label {
        font-family: 'Manrope', sans-serif;
    }
    
    /* Buttons */
    .stButton button {
        font-family: 'Manrope', sans-serif;
    }
    
    /* Selectbox and other inputs */
    .stSelectbox label, .stTextArea label, .stMetric label {
        font-family: 'Manrope', sans-serif;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        font-family: 'Manrope', sans-serif;
    }
    
    /* Code blocks - you might want to keep these as monospace */
    .stCode {
        font-family: 'Manrope', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

def reset_analysis_state():
    """Reset all analysis-related session state"""
    st.session_state.agent_context = AgentContext()
    st.session_state.analysis_complete = False
    st.session_state.agent = None
    st.session_state.results = {}
    st.session_state.preliminary_analysis = None  # Add this for sentiment analysis
    st.session_state.diarised_transcript = None  # Add this for tagged dialogue
    if 'previous_transcript' in st.session_state:
        del st.session_state.previous_transcript


def main():
    apply_custom_css()
    st.title("Customer Service Analysis Agent")
    st.markdown("---")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Transcript Input")
        transcript_content = ""
        
        # Point this to your conversations directory
        transcripts_dir = "paysafe/conversations"
        transcripts = load_transcript_files(transcripts_dir)
        
        if transcripts:
            transcript_names = list(transcripts.keys())
            
            # Dropdown to select conversation
            selected_file = st.selectbox("Select a conversation:", transcript_names)
            
            # **KEY FIX: Check if transcript changed and reset state**
            if 'previous_transcript' not in st.session_state:
                st.session_state.previous_transcript = selected_file
            elif st.session_state.previous_transcript != selected_file:
                # Transcript changed - reset everything
                reset_analysis_state()
                st.session_state.previous_transcript = selected_file
                st.rerun()  # Force rerun to clear the UI
            
            transcript_content = transcripts[selected_file]["content"]
            
            if selected_file:
                st.subheader(f"Selected Conversation: {selected_file}")
                st.text_area(
                    "Transcript Content",
                    transcripts[selected_file]["content"],
                    height=400,
                )
        else:
            st.warning("No transcripts found in the directory.")
    
    with col2:
        st.header("Preliminary Analysis")
        
        # Button to run preliminary analysis
        if st.button("🔍 Run Preliminary Analysis", type="secondary"):
            if not transcript_content:
                st.error("Please select a transcript first!")
            else:
                with st.spinner("Running preliminary analysis..."):
                    try:
                        # Import and initialize the sentiment analyzer
                        from sentiment_analyzer import ConversationSummarizer, summary_to_dataframe, style_sentiment_dataframe
                        
                        # Initialize the summarizer
                        summarizer = ConversationSummarizer()
                        
                        # Tag the dialogue
                        diarised_text = summarizer.tag_dialogue_with_llm(transcript_content)
                        st.session_state.diarised_transcript = diarised_text
                        
                        # Process the conversation
                        summary = summarizer.process_conversation(transcript_content)
                        st.session_state.preliminary_analysis = summary
                        
                        st.success("✅ Preliminary analysis completed!")
                        
                    except Exception as e:
                        st.error(f"Error during preliminary analysis: {str(e)}")
                        st.exception(e)
        
        # Display tabs
        tab1, tab2 = st.tabs(["Diarised Outputs", "Data Extraction"])
        
        with tab1:
            st.subheader("Tagged Conversation")
            
            if st.session_state.get('diarised_transcript'):
                st.text_area(
                    "Diarised Output",
                    value=st.session_state.diarised_transcript,
                    height=400,
                    disabled=True
                )
            else:
                st.info("Click 'Run Preliminary Analysis' to generate diarised output.")
        
        with tab2:
            st.subheader("Extracted Metrics")
            
            if st.session_state.get('preliminary_analysis'):
                try:
                    from sentiment_analyzer import summary_to_dataframe, style_sentiment_dataframe
                    
                    summary = st.session_state.preliminary_analysis
                    
                    # Convert summary to dataframes
                    dataframes = summary_to_dataframe(summary)
                    
                    # Display each DataFrame
                    for name, df in dataframes.items():
                        st.markdown(f"### {name}")
                        
                        if not df.empty:
                            styled_df = style_sentiment_dataframe(df)
                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            st.info(f"No data available for {name}")
                        
                        st.markdown("---")  # Add separator between sections
                        
                except Exception as e:
                    st.error(f"Error displaying metrics: {str(e)}")
                    st.exception(e)
            else:
                st.info("Click 'Run Preliminary Analysis' to extract metrics.")
    
    # Analysis section
    st.markdown("---")
    st.header("Deep Analysis")
    
    # Add two columns for buttons
    btn_col1, btn_col2 = st.columns([3, 1])
    
    with btn_col1:
        if st.button("🚀 Start Deep Analysis", type="primary"):
            if not transcript_content:
                st.error("Please select a transcript first!")
            else:
                with st.spinner("Initializing deep analysis..."):
                    # Initialize LLM and data
                    llm = initialize_llm()
                    if not llm:
                        return
                    
                    collated_engine, transaction_engine = load_data_indexes()
                    if not collated_engine or not transaction_engine:
                        st.error("Error loading data indexes. Please check your data files.")
                        return
                    
                    # Initialize context
                    st.session_state.agent_context.initialize(transcript_content)
                    
                    # Create tools and agent
                    tools = create_tools(llm, collated_engine, transaction_engine, st.session_state.agent_context)
                    
                    agent_context_prompt = """
You are an autonomous Customer Service Analysis Agent. Your job is to analyze transcripts and identify root causes behind customer issues.

Follow this strict tool usage order:
1. ALWAYS call 'analyze_content' FIRST to extract name, issue, and status.
2. Then call 'check_context_status'. If the issue is resolved, perform RCA for learning purpose.
3. If RCA is needed, call 'decide_context_tool' to determine what data is required.
4. Based on that decision:
   - Call 'run_conversation_tool' if only conversation context is needed.
   - Call BOTH 'run_conversation_tool' and 'run_transaction_tool' if full context is needed.
5. ONLY AFTER the proper context is available, call 'perform_root_cause_analysis'
6. After RCA, call 'recommendation_retreiver' to gather relevant context for recommendations.
7. Finally, call 'recommendation' to generate actionable recommendations based on the issue and context.

NEVER call both 'perform_root_cause_analysis' and 'timeline_analysis' without first ensuring the required context is loaded.

You are fully autonomous. Use tools as needed, rerun steps when necessary, and ensure your insights drive measurable improvements in customer satisfaction and operational efficiency.
                    """
                    
                    agent = ReActAgent(
                        llm=llm,
                        tools=tools,
                        memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
                        max_iterations=50,
                        context=agent_context_prompt,
                        verbose=True
                    )
                    
                    st.session_state.agent = agent
                
                # Run analysis
                with st.spinner("Running deep analysis... This may take a few minutes."):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Run the agent
                        message = """
I have a transcript of a customer service interaction that has been loaded into the context. 
Please follow this workflow:

Please execute each step methodically and report on your progress at each stage. 
Use the tools provided and follow the workflow exactly as described.
                        """
                        
                        status_text.text("Starting analysis...")
                        progress_bar.progress(10)
                        
                        response = agent.chat(message)
                        
                        progress_bar.progress(100)
                        status_text.text("Analysis completed!")
                        
                        st.session_state.results['response'] = str(response)
                        st.session_state.analysis_complete = True
                        
                        st.success("✅ Deep analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        st.exception(e)
    
    with btn_col2:
        if st.button("🔄 Reset All", help="Clear all analysis and start fresh"):
            reset_analysis_state()
            st.rerun()
    
    # Results section
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.header("Deep Analysis Results")
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Summary", 
            "🔍 Root Cause Analysis", 
            "💡 Recommendations", 
            "📅 Timeline"
        ])
        
        with tab1:
            st.subheader("Extracted Information")
            context_data = st.session_state.agent_context.get_all()
            
            if context_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'name' in context_data:
                        st.metric("Customer Name", context_data['name'])
                    if 'issue' in context_data:
                        st.metric("Main Issue", context_data['issue'])
                
                with col2:
                    if 'status' in context_data:
                        st.metric("Status", context_data['status'])
                    if 'sentiment' in context_data:
                        st.metric("Sentiment", context_data['sentiment'])
                
                if 'summary' in context_data:
                    st.subheader("Summary")
                    st.write(context_data['summary'].replace('$', '\\$'))
        
        with tab2:
            context_data = st.session_state.agent_context.get_all()
            
            if 'rootcause' in context_data and context_data['rootcause']:
                rca_content = context_data['rootcause']
                parsed_rca = parse_rca_response_simple(rca_content)
                
                if parsed_rca:
                    if 'rootcause' in parsed_rca:
                        st.write("**Root Cause:**")
                        st.write(parsed_rca['rootcause'].replace('$', '\\$'))
                        st.markdown("---")
                    
                    if 'pattern' in parsed_rca:
                        st.write("**Patterns:**")
                        st.write(parsed_rca['pattern'].replace('$', '\\$'))
                        st.markdown("---")
                    
                    if 'rationale' in parsed_rca:
                        st.write("**Rationale:**")
                        st.write(parsed_rca['rationale'].replace('$', '\\$'))
                else:
                    # Fallback: display raw RCA content
                    st.write("**Root Cause Analysis:**")
                    st.write(rca_content.replace('$', '\\$'))
            else:
                st.info("Root cause analysis not available. Please ensure the analysis completed successfully.")
        
        with tab3:
            st.subheader("Recommendations")
            context_data = st.session_state.agent_context.get_all()
            
            # Check if we have recommendation data stored in the context
            if 'recommendation' in context_data or 'recommendations' in context_data:
                recommendation_content = context_data.get('recommendation') or context_data.get('recommendations')
                
                if recommendation_content:
                    # Parse and display recommendations
                    if "RECOMMENDATIONS:" in recommendation_content:
                        recommendations_text = recommendation_content.split("RECOMMENDATIONS:")[1].strip()
                        st.write(recommendations_text)
                    else:
                        st.write(recommendation_content)
                else:
                    st.info("Recommendations are being generated...")
            else:
                # Check if we have retrieved information for recommendations
                if 'retrieved_information' in context_data:
                    st.info("Context retrieved for recommendations. Recommendations should be generated in the next step.")
                else:
                    st.info("Recommendations not yet available. Please ensure the complete analysis workflow is executed.")
        
        with tab4:
            try:
                # Generate events
                events = generate_events(transcript_content)
                
                if not events or not events.strip():
                    st.warning("⚠️ No timeline events could be generated from the transcript.")
                else:
                    # Parse timeline data
                    timeline_data = [line.strip() for line in events.split('\n') if line.strip()]
                    
                    if not timeline_data:
                        st.warning("⚠️ Could not parse timeline events.")
                    else:
                        # Show raw timeline text in an expander for debugging
                        with st.expander("📝 View Raw Timeline Text"):
                            st.text(events)
                        
                        # Create the vertical timeline visualization
                        result = create_matplotlib_vertical_timeline(timeline_data)
                        
                        if result:
                            fig, ax = result
                            st.pyplot(fig)
                            plt.close(fig)  # Clean up to prevent memory leaks
                        else:
                            st.error("❌ Failed to create timeline visualization.")
                            
            except Exception as e:
                st.error(f"❌ Error generating timeline: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.exception(e)
        
        # Export functionality
        st.markdown("---")
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Deep Analysis Results"):
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "transcript_file": st.session_state.get('previous_transcript', 'unknown'),
                    "extracted_data": st.session_state.agent_context.get_all(),
                    "raw_response": st.session_state.results.get('response', ''),
                }
                
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="💾 Download Deep Analysis JSON",
                    data=json_str,
                    file_name=f"deep_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.session_state.get('preliminary_analysis'):
                if st.button("📥 Export Preliminary Analysis"):
                    prelim_export = {
                        "timestamp": datetime.now().isoformat(),
                        "transcript_file": st.session_state.get('previous_transcript', 'unknown'),
                        "diarised_transcript": st.session_state.get('diarised_transcript', ''),
                        "analysis": st.session_state.preliminary_analysis
                    }
                    
                    json_str = json.dumps(prelim_export, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="💾 Download Preliminary Analysis JSON",
                        data=json_str,
                        file_name=f"preliminary_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )


if __name__ == "__main__":
    main()
