import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from neo4j import GraphDatabase
import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv

# --- CONFIGURATION ---
st.set_page_config(page_title="Bridge Maintenance AI Dashboard", layout="wide")

# Settings come from a .env file in the repository root (see .env.example).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Configure Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- NEO4J CONNECTION ---
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_query(query, params=None):
    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]

# --- DATA LOADERS ---
@st.cache_data
def load_all_bridges_for_map():
    """Fetch basic location data for ALL bridges for the map."""
    query = """
    MATCH (b:Bridge)-[:LOCATED_IN]->(l:Location)
    WHERE l.latitude IS NOT NULL AND l.longitude IS NOT NULL
    RETURN b.structureNumber as id, l.latitude as lat, l.longitude as lon, 
           l.countyName as county, b.yearBuilt as year
    // REMOVED LIMIT to show all bridges
    """
    return pd.DataFrame(run_query(query))

def get_bridge_details(structure_number):
    """Fetch detailed snapshot data for a single bridge."""
    query = """
    MATCH (b:Bridge {structureNumber: $id})
    OPTIONAL MATCH (b)-[:LOCATED_IN]->(l:Location)
    OPTIONAL MATCH (b)-[:HAS_TRAFFIC_PROFILE]->(t:TrafficProfile)
    OPTIONAL MATCH (l)-[:HAS_EXPOSURE]->(w:WeatherExposure)
    RETURN b, l, t, w
    """
    data = run_query(query, {"id": structure_number})
    return data[0] if data else None

def get_element_history(structure_number):
    """Fetch element condition history over time."""
    query = """
    MATCH (b:Bridge {structureNumber: $id})-[:HAS_ELEMENT]->(e:BridgeElement)
    // Renamed 'Quantity' to 'TotalQty' to avoid conflict with melt()
    RETURN e.elementName as Element, e.year as Year, e.totalQuantity as TotalQty, 
           e.cs1 as CS1, e.cs2 as CS2, e.cs3 as CS3, e.cs4 as CS4
    ORDER BY e.year, e.elementName
    """
    return pd.DataFrame(run_query(query, {"id": structure_number}))

# --- AI AGENT FUNCTIONS ---
def get_cypher_from_llm(user_query):
    """Uses Gemini to translate natural language to Cypher."""
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    
    # Improved Schema Summary to help the AI write better queries
    schema_summary = """
    Nodes & Properties:
    - (:Bridge {structureNumber (string), yearBuilt (int), bridgeAge (int), mainSpanMaterial (string), structureLength (float), deckArea (float), yearReconstructed (int), scourCritical (string)})
    - (:Location {name (string, e.g., "Tulsa County, Oklahoma"), countyName (string)})
    - (:BridgeElement {elementName (string), year (int), cs1 (float), cs2 (float), cs3 (float), cs4 (float), hi (float)})
    - (:Inspection {inspectionDate (string), nbiYear (int)})
    - (:WeatherExposure {freezeThawCycles (int)})
    - (:TrafficProfile {adt (int), adttVolume (int)})
    
    Relationships:
    - (:Bridge)-[:LOCATED_IN]->(:Location)
    - (:Bridge)-[:HAS_ELEMENT]->(:BridgeElement)
    - (:Bridge)-[:HAS_TRAFFIC_PROFILE]->(:TrafficProfile)
    - (:Location)-[:HAS_EXPOSURE]->(:WeatherExposure)
    
    IMPORTANT TIPS FOR CYPHER GENERATION:
    1. Location names are stored as "County Name, State Name" (e.g., "Tulsa County, Oklahoma"). 
       ALWAYS use 'CONTAINS' for location searches. Example: WHERE l.name CONTAINS 'Tulsa'.
    2. Property names are case-sensitive. Use exact names like 'structureNumber', 'yearBuilt', 'bridgeAge'.
    3. Return limits: Always limit results to 50 unless asked otherwise.
    """
    
    prompt = f"""
    You are a Neo4j Cypher expert. Convert this question into a Cypher query for the schema below.
    Schema: {schema_summary}
    Question: "{user_query}"
    IMPORTANT: Return ONLY the raw Cypher query text. No markdown, no explanations.
    """
    response = model.generate_content(prompt)
    return response.text.strip().replace("```cypher", "").replace("```", "")

def get_natural_language_response(user_query, db_results):
    """Uses Gemini to summarize database results."""
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    prompt = f"""
    User Question: "{user_query}"
    Database Results: {db_results}
    
    Please answer the user's question clearly based ONLY on the data provided. 
    If the data is a list of bridges, summarize it (e.g., "I found 5 bridges matching your criteria...").
    """
    response = model.generate_content(prompt)
    return response.text

# --- DASHBOARD UI ---

st.title("🌉 Bridge Maintenance AI Dashboard")

# Sidebar for Navigation
page = st.sidebar.radio("Navigate", ["Map View", "Bridge Snapshot & History", "AI Assistant", "Forecasting"])

# --- PAGE 1: MAP VIEW ---
if page == "Map View":
    st.header("Network Overview")
    
    with st.spinner("Loading map data (this may take a moment)..."):
        map_data = load_all_bridges_for_map()
    
    if not map_data.empty:
        # PyDeck Map with LARGER DOTS
        layer = pdk.Layer(
            "ScatterplotLayer",
            map_data,
            get_position=["lon", "lat"],
            get_color=[200, 30, 0, 160],
            get_radius=500,          # Radius in meters
            pickable=True,
            radius_min_pixels=5,     # Minimum pixel size (crucial for visibility when zoomed out)
            radius_max_pixels=20,
        )
        
        view_state = pdk.ViewState(
            latitude=map_data["lat"].mean(),
            longitude=map_data["lon"].mean(),
            zoom=7,
            pitch=0,
        )
        
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[layer],
            tooltip={"text": "Bridge: {id}\nCounty: {county}\nYear Built: {year}"}
        ))
        
        st.metric("Total Bridges Loaded", len(map_data))
    else:
        st.warning("No location data found. Check your Neo4j database.")

# --- PAGE 2: BRIDGE SNAPSHOT & HISTORY ---
elif page == "Bridge Snapshot & History":
    st.header("Bridge Inspector")
    
    with st.spinner("Loading bridge list..."):
        bridge_list = load_all_bridges_for_map()['id'].tolist()
        
    selected_bridge_id = st.selectbox("Select a Bridge ID:", bridge_list)
    
    if selected_bridge_id:
        details = get_bridge_details(selected_bridge_id)
        
        if details:
            b = details['b']
            l = details.get('l', {})
            w = details.get('w', {})
            t = details.get('t', {})
            
            # 1. Snapshot Board
            st.subheader(f"Snapshot: Bridge {selected_bridge_id}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Year Built", b.get('yearBuilt', 'N/A'))
            col2.metric("County", l.get('countyName', 'N/A'))
            col3.metric("Daily Traffic (ADT)", t.get('adt', 'N/A'))
            col4.metric("Freeze-Thaw Cycles", w.get('freezeThawCycles', 'N/A'))
            
            # 2. Health History Chart
            st.subheader("Element Health History (Last 5 Years)")
            hist_df = get_element_history(selected_bridge_id)
            
            if not hist_df.empty:
                # FIX: Melt the DataFrame to make it 'long-form' for Plotly
                # Changed value_name="Quantity" to "Element Quantity" just to be safe
                long_df = hist_df.melt(
                    id_vars=["Year", "Element"], 
                    value_vars=["CS1", "CS2", "CS3", "CS4"],
                    var_name="Condition State", 
                    value_name="Element Quantity" 
                )
                
                fig = px.bar(long_df, x="Year", y="Element Quantity", 
                             color="Condition State",
                             title="Element Condition States Over Time",
                             facet_col="Element", facet_col_wrap=2,
                             color_discrete_map={"CS1": "green", "CS2": "yellow", "CS3": "orange", "CS4": "red"})
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Raw Element Data"):
                    st.dataframe(hist_df)
            else:
                st.info("No element history data available for this bridge.")

# --- PAGE 3: AI ASSISTANT ---
elif page == "AI Assistant":
    st.header("🤖 Graph AI Assistant")
    st.markdown("Ask questions like: *'Show me bridges in Tulsa county older than 1980'*")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the bridge network..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            try:
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                # 1. Get Query
                cypher_query = get_cypher_from_llm(prompt)
                
                # 2. Display Query (Debugging/Transparency)
                st.code(cypher_query, language="cypher")
                
                # 3. Run Query
                results = run_query(cypher_query)
                
                # 4. Handle Empty Results gracefully
                if not results:
                    final_answer = "I generated a query, but the database returned no results. Try rephrasing (e.g., check the county name spelling)."
                else:
                    truncated_results = results[:20] 
                    final_answer = get_natural_language_response(prompt, truncated_results)
                
                message_placeholder.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.error(error_msg)

# --- PAGE 4: FORECASTING (Placeholder) ---
elif page == "Forecasting":
    st.header("🔮 Deterioration Forecasting")
    st.info("This module will use Machine Learning to predict future condition ratings based on historical NBE data, traffic, and weather.")
    
    st.subheader("Model Inputs")
    st.write("- Historical Element Conditions (CS1-CS4)")
    st.write("- Age & Material")
    st.write("- Environmental Stressors (Freeze/Thaw, Precip)")
    
    st.button("Run Prediction Model (Coming Soon)")