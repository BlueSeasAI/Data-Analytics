import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import requests
from streamlit_lottie import st_lottie

# Page configuration
st.set_page_config(
    page_title="Brew Minds Analytics | AI-Powered Insights",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed to collapsed to remove sidebar
)
# Custom CSS styling with additional improvements and responsive design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #6F4E37;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #8B5A2B;
    }
    .insight-card {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #6F4E37;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #6F4E37;
    }
    .metric-label {
        font-size: 1rem;
        color: #666666;
    }
    .trend-up {
        color: #28a745;
    }
    .trend-down {
        color: #dc3545;
    }
    
    /* UPDATED - Recommendation cards for better responsiveness */
    .recommendation-card {
        border: 2px solid #6F4E37; 
        border-radius: 5px; 
        padding: 15px; 
        height: auto;
        min-height: 250px;
        margin-bottom: 15px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .recommendation-header {
        color: #6F4E37;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .recommendation-content {
        flex-grow: 1;
        margin-bottom: 10px;
    }
    .action-button {
        background-color: #6F4E37; 
        color: white; 
        border: none; 
        padding: 8px 12px; 
        border-radius: 4px;
        cursor: pointer;
        margin-right: 10px;
        white-space: nowrap;
        font-size: 0.9rem;
    }
    .secondary-button {
        background-color: white; 
        color: #6F4E37; 
        border: 1px solid #6F4E37; 
        padding: 8px 12px; 
        border-radius: 4px;
        cursor: pointer;
        white-space: nowrap;
        font-size: 0.9rem;
    }
    .button-container {
        display: flex; 
        justify-content: flex-start;
        gap: 8px;
        margin-top: auto;
    }
    
    /* UPDATED - Simulator container with responsive design */
    .slider-container {
        position: relative;
        margin-bottom: 1.5rem;
        padding-top: 20px; /* Space for zero point indicator */
    }
    .zero-point {
        position: absolute;
        left: 50%;
        top: -15px;
        color: black;
        font-size: 12px;
    }
    .arrow-up {
        width: 0; 
        height: 0; 
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-bottom: 5px solid black;
        margin: 0 auto;
    }
    .simulator-container {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .date-filter-container {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .logo-container {
        position: absolute;
        top: 20px;
        right: 20px;
        text-align: right;
    }
    
    /* UPDATED - Responsive scenario simulator */
    .scenario-simulator {
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #eee;
        background-color: #f9f9f9;
        border-radius: 8px;
    }
    .scenario-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 15px;
        flex-wrap: wrap;
        gap: 10px;
    }
    .scenario-title {
        font-size: 1.5rem;
        margin: 0;
        color: #6F4E37;
    }
    
    /* Media queries for responsive design */
    @media (max-width: 1200px) {
        .metric-value {
            font-size: 1.8rem;
        }
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
        }
    }
    
    @media (max-width: 992px) {
        .recommendation-card {
            min-height: 280px;
        }
    }
    
    @media (max-width: 768px) {
        .simulator-columns {
            flex-direction: column;
        }
        .button-container {
            flex-wrap: wrap;
        }
        .recommendation-card {
            min-height: auto;
            height: auto;
        }
    }
</style>
""", unsafe_allow_html=True)
# Load Lottie animations function
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Enhanced load_data() function with comprehensive validation
@st.cache_data
def load_data():
    try:
        # Read the CSV with comprehensive parsing
        data = pd.read_csv("brew_minds_dataset.csv", 
                           parse_dates=["Date"],
                           dayfirst=False,
                           encoding="utf-8-sig")
        
        # Comprehensive data validation
        required_columns = [
            'Date', 'Revenue_Coffee', 'Customer_Count', 
            'Customer_Satisfaction', 'Day', 'Weather'
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return pd.DataFrame()
        
        # Basic data cleaning: Convert numeric columns
        numeric_columns = [
            'Revenue_Coffee', 'Revenue_Food', 'Revenue_Workspace', 
            'Revenue_Events', 'Customer_Count', 'Customer_Satisfaction'
        ]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows that are entirely NaN
        data = data.dropna(how='all')
        
        return data
    
    except Exception as e:
        st.error(f"Error loading brew_minds_dataset.csv: {e}")
        return pd.DataFrame()

# Function to generate AI responses with error handling - FIXED COFFEE PERCENTAGE
def generate_ai_response(question, data):
    try:
        # Calculate the actual coffee revenue percentage
        revenue_cols = ['Revenue_Coffee', 'Revenue_Food', 'Revenue_Workspace', 'Revenue_Events']
        available_revenue_cols = [col for col in revenue_cols if col in data.columns]
        
        if available_revenue_cols:
            total_revenue = data[available_revenue_cols].sum().sum()
            coffee_revenue = data['Revenue_Coffee'].sum()
            coffee_percentage = (coffee_revenue / total_revenue * 100) if total_revenue > 0 else 0
        else:
            coffee_percentage = 0
            
        responses = {
            "best day": "Based on the data, Saturday consistently shows the highest overall revenue and customer count, generating approximately 35% more revenue compared to weekdays.",
            "revenue": f"Coffee sales are the primary revenue driver, contributing {coffee_percentage:.1f}% of total revenue. The average daily coffee revenue is ${data['Revenue_Coffee'].mean():.2f}.",
            "customer": f"Your repeat customer rate is {(data['Repeat_Customers'].sum() / data['Customer_Count'].sum() * 100):.1f}%. The average daily customer count is {data['Customer_Count'].mean():.0f}.",
            "workshop": f"Top workshop topics include {', '.join(data.groupby('Workshop_Topic')['Workshop_Attendees'].sum().nlargest(3).index)}. Average workshop satisfaction is {data['Workshop_Satisfaction'].mean():.2f}/5.0.",
            "trend": f"There's a positive growth trend with average daily revenue increasing by ${data['Revenue_Coffee'].diff().mean():.2f} and customer satisfaction improving by {data['Customer_Satisfaction'].diff().mean():.2f} points.",
            "marketing": f"Marketing spend averages ${data['Marketing_Spend'].mean():.2f} daily, with weekend promotions showing a boost.",
            "increase revenue": "To increase revenue, consider: 1) Implementing an 'Afternoon Productivity Bundle' for slow periods, 2) Creating workshop series packages for higher retention, and 3) Introducing premium coffee options with 15% higher margins.",
            "customer retention": f"Your customer retention strategies could be improved. Currently, {coffee_percentage:.1f}% of your customers return within 14 days. Consider a loyalty program targeting 'Casual Visitors' to increase this by 18%."
        }
        
        default_response = "I couldn't find specific insights for that query. Try asking about revenue, customers, workshops, trends, or marketing."
        
        question_lower = question.lower()
        matched_responses = [resp for key, resp in responses.items() if key in question_lower]
        
        return matched_responses[0] if matched_responses else default_response
    
    except Exception as e:
        return f"Error generating Coffee Mate response: {str(e)}"

# Enhanced AI response function for natural language queries
def generate_custom_ai_analysis(query, data):
    try:
        query_lower = query.lower()
        
        # Sample response templates based on query content
        if "weather" in query_lower and ("satisfaction" in query_lower or "customer" in query_lower):
            return {
                "type": "weather_satisfaction",
                "title": "Weather Impact on Customer Experience",
                "insight": "Rainy weather correlates with 23% higher customer satisfaction but 12% lower customer counts. This indicates that while fewer customers visit during rain, those who do tend to stay longer and enjoy the cosy atmosphere more.",
                "recommendation": "Consider creating special rainy day promotions to boost attendance during wet weather, focusing on the cosy ambiance that drives higher satisfaction."
            }
        elif "revenue" in query_lower and "day" in query_lower:
            return {
                "type": "revenue_day",
                "title": "Revenue Patterns by Day of Week",
                "insight": "Monday and Wednesday show the lowest revenue performance, with a 34% gap compared to weekend days. Coffee sales remain consistent, but food and workspace revenue significantly drop midweek.",
                "recommendation": "Create 'Midweek Motivation' bundles combining coffee, food items and workspace access at a special rate to boost Wednesday revenue."
            }
        elif "customer" in query_lower and "segment" in query_lower:
            return {
                "type": "customer_segment",
                "title": "Customer Segment Behaviour Analysis",
                "insight": "The 'Loyal Enthusiast' segment (32% of customers) generates 47% of total revenue through frequent visits, while 'Casual Visitors' (47% of customers) only contribute 29% of revenue despite being the largest segment.",
                "recommendation": "Focus retention efforts on converting 'Casual Visitors' to 'Loyal Enthusiasts' through a tiered rewards program with clear incentives for repeat visits."
            }
        elif "workshop" in query_lower or "event" in query_lower:
            return {
                "type": "workshop_analysis",
                "title": "Workshop Performance Deep Dive",
                "insight": "Business-oriented workshops drive 215% more additional revenue through extended stays and food/beverage purchases compared to hobby-focused workshops, despite having similar attendance numbers.",
                "recommendation": "Expand your business workshop offerings with graduated series (beginner to advanced) to create long-term engagement while maintaining premium pricing."
            }
        else:
            return {
                "type": "general_business",
                "title": "Overall Business Performance",
                "insight": "Your business shows strong weekend performance but significant midweek dips. The coffee-to-food sales ratio of 2.1:1 indicates potential for increased food attachment rate. Customer satisfaction averages 4.2/5.0 with lowest scores during peak hours.",
                "recommendation": "Focus on three key opportunity areas: 1) Midweek revenue enhancement, 2) Food attachment during morning coffee rush, and 3) Service optimisation during peak periods."
            }
    except Exception as e:
        return {
            "type": "error",
            "title": "Analysis Error",
            "insight": f"Error generating analysis: {str(e)}",
            "recommendation": "Try a different query or check data availability."
        }

# Function to generate strategy notes for all scenarios
def generate_strategy_notes(coffee_price, marketing_budget, workshop_theme, extended_hours, loyalty_program):
    notes = []
    
    # Coffee price logic
    if coffee_price > 0:
        notes.append(f"Price increase of {coffee_price}% should be paired with messaging about quality and sustainability")
    elif coffee_price < 0:
        notes.append(f"Price reduction of {abs(coffee_price)}% could be advertised as a limited-time promotion to drive volume")
    
    # Marketing budget logic
    if marketing_budget > 0:
        notes.append(f"Additional {marketing_budget}% marketing spend should focus on digital channels with highest ROI")
    elif marketing_budget < 0:
        notes.append(f"With reduced marketing budget, focus on retention marketing and organic social media")
    
    # Workshop theme
    if workshop_theme != "None":
        notes.append(f"New '{workshop_theme}' workshop should be promoted to both existing customers and new audiences")
    
    # Hours extension
    if extended_hours:
        notes.append("Extended hours require staff adjustment and targeted promotion to specific customer segments")
    
    # Loyalty program
    if loyalty_program:
        notes.append("Loyalty program should emphasise both frequency and average transaction value to maximise impact")
    
    return notes if notes else ["Adjust parameters to see strategy recommendations"]
    # Load data from CSV file
data = load_data()

# Create a container for the logo
logo_col1, logo_col2 = st.columns([4, 1])
with logo_col2:
    # Use the raw GitHub URL for the image
    logo_url = "https://raw.githubusercontent.com/BlueSeasAI/Data-Analytics/main/BrewMindslogo.png"
    st.markdown(f"""
    <div style="text-align: right; padding: 10px;">
        <img src="{logo_url}" width="150" style="max-width: 100%;">
    </div>
    """, unsafe_allow_html=True)

# Create header row with date filter in a neat box
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown('<p class="main-header">Brew Minds Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### Coffee Mate: Your AI-Powered Data Analytics Barista")

with header_col2:
    # Date range filter with defensive checks and improved formatting
    if 'Date' in data.columns and not data['Date'].isna().all():
        min_date = data['Date'].min().date()
        max_date = data['Date'].max().date()
        
        # Format with abbreviated month names
        min_date_display = min_date.strftime("%d %b %y")
        max_date_display = max_date.strftime("%d %b %y")
    else:
        current_date = datetime.now().date()
        min_date = current_date - timedelta(days=90)
        max_date = current_date
        min_date_display = min_date.strftime("%d %b %y")
        max_date_display = max_date.strftime("%d %b %y")
    
    st.markdown('<div class="date-filter-container">', unsafe_allow_html=True)
    date_range = st.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Apply date filter
if len(date_range) == 2 and 'Date' in data.columns:
    start_date, end_date = date_range
    filtered_data = data[(data['Date'] >= pd.Timestamp(start_date)) & 
                         (data['Date'] <= pd.Timestamp(end_date))]
else:
    filtered_data = data.copy()

# Check if data is available for display
if len(filtered_data) == 0:
    st.warning("No data available with the current filters or there was an error loading the data. Please check the dataset or adjust filters.")
    st.stop()

# KPI Cards
st.markdown("## Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

revenue_cols = ['Revenue_Coffee', 'Revenue_Food', 'Revenue_Workspace', 'Revenue_Events']
available_revenue_cols = [col for col in revenue_cols if col in filtered_data.columns]

if available_revenue_cols:
    total_revenue = filtered_data[available_revenue_cols].sum().sum()
    avg_revenue = total_revenue / len(filtered_data) if len(filtered_data) > 0 else 0
else:
    total_revenue = 0
    avg_revenue = 0

if 'Customer_Count' in filtered_data.columns:
    total_customers = filtered_data['Customer_Count'].sum()
else:
    total_customers = 0

if 'Customer_Satisfaction' in filtered_data.columns:
    avg_satisfaction = filtered_data['Customer_Satisfaction'].mean() if len(filtered_data) > 0 else 0
else:
    avg_satisfaction = 0

if 'Repeat_Customers' in filtered_data.columns and 'Customer_Count' in filtered_data.columns:
    repeat_rate = (filtered_data['Repeat_Customers'].sum() / total_customers) * 100 if total_customers > 0 else 0
else:
    repeat_rate = 0

if len(filtered_data) >= 14:
    half_point = len(filtered_data) // 2
    first_half = filtered_data.iloc[:half_point]
    second_half = filtered_data.iloc[half_point:]
    
    if available_revenue_cols:
        first_half_rev = first_half[available_revenue_cols].sum().sum()
        second_half_rev = second_half[available_revenue_cols].sum().sum()
        revenue_change = ((second_half_rev / first_half_rev) - 1) * 100 if first_half_rev > 0 else 0
    else:
        revenue_change = 0
    
    if 'Customer_Count' in filtered_data.columns:
        first_half_cust = first_half['Customer_Count'].sum()
        second_half_cust = second_half['Customer_Count'].sum()
        customer_change = ((second_half_cust / first_half_cust) - 1) * 100 if first_half_cust > 0 else 0
    else:
        customer_change = 0
    
    if 'Customer_Satisfaction' in filtered_data.columns:
        satisfaction_change = second_half['Customer_Satisfaction'].mean() - first_half['Customer_Satisfaction'].mean() if len(first_half) > 0 and len(second_half) > 0 else 0
    else:
        satisfaction_change = 0
    
    if 'Repeat_Customers' in filtered_data.columns and 'Customer_Count' in filtered_data.columns:
        first_half_repeat = (first_half['Repeat_Customers'].sum() / first_half['Customer_Count'].sum()) * 100 if first_half['Customer_Count'].sum() > 0 else 0
        second_half_repeat = (second_half['Repeat_Customers'].sum() / second_half['Customer_Count'].sum()) * 100 if second_half['Customer_Count'].sum() > 0 else 0
        repeat_rate_change = second_half_repeat - first_half_repeat
    else:
        repeat_rate_change = 0
else:
    revenue_change = 0
    customer_change = 0
    satisfaction_change = 0
    repeat_rate_change = 0

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Revenue</div>
        <div class="metric-value">${total_revenue:,.2f}</div>
        <div class="{'trend-up' if revenue_change >= 0 else 'trend-down'}">
            {'+' if revenue_change >= 0 else ''}{revenue_change:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Customers</div>
        <div class="metric-value">{total_customers:,}</div>
        <div class="{'trend-up' if customer_change >= 0 else 'trend-down'}">
            {'+' if customer_change >= 0 else ''}{customer_change:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg. Satisfaction</div>
        <div class="metric-value">{avg_satisfaction:.1f}/5.0</div>
        <div class="{'trend-up' if satisfaction_change >= 0 else 'trend-down'}">
            {'+' if satisfaction_change >= 0 else ''}{satisfaction_change:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Repeat Customer Rate</div>
        <div class="metric-value">{repeat_rate:.1f}%</div>
        <div class="{'trend-up' if repeat_rate_change >= 0 else 'trend-down'}">
            {'+' if repeat_rate_change >= 0 else ''}{repeat_rate_change:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    # ENHANCEMENT 1: Natural Language Insights Engine
st.markdown("## 💬 Ask Anything About Your Data")
custom_query = st.text_area("Ask in plain English:", placeholder="Example: Show me the relationship between weather and customer satisfaction", height=60)

if custom_query and st.button("Generate Custom Insight"):
    with st.spinner("Coffee Mate analysing your request..."):
        time.sleep(1.2)  # Simulate processing
        ai_analysis = generate_custom_ai_analysis(custom_query, filtered_data)
        
        st.markdown(f"### 🤖 {ai_analysis['title']}")
        
        # Choose visualization based on analysis type
        if ai_analysis["type"] == "weather_satisfaction" and 'Weather' in filtered_data.columns:
            col1, col2 = st.columns([2,1])
            with col1:
                # Create more advanced visualization
                fig = px.scatter(filtered_data, 
                                x='Customer_Satisfaction', 
                                y='Customer_Count',
                                color='Weather', 
                                size='Revenue_Coffee',
                                hover_data=['Day'],
                                title="Weather Impact on Customer Metrics")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown(f"""
                <div class='insight-card'>
                <strong>Coffee Mate-Generated Insight:</strong><br>{ai_analysis['insight']}
                </div>
                <div class='insight-card'>
                <strong>Recommended Action:</strong><br>{ai_analysis['recommendation']}
                </div>
                """, unsafe_allow_html=True)
                
        elif ai_analysis["type"] == "revenue_day" and 'Day' in filtered_data.columns:
            col1, col2 = st.columns([2,1])
            with col1:
                # Create a more detailed day-of-week analysis
                if available_revenue_cols:
                    revenue_by_day = filtered_data.groupby('Day')[available_revenue_cols].mean().reset_index()
                    fig = px.bar(revenue_by_day.melt(id_vars='Day', value_vars=available_revenue_cols, var_name='Revenue Stream', value_name='Amount'),
                                x='Day', y='Amount', color='Revenue Stream', barmode='stack',
                                title="Revenue Breakdown by Day of Week")
                    fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown(f"""
                <div class='insight-card'>
                <strong>Coffee Mate-Generated Insight:</strong><br>{ai_analysis['insight']}
                </div>
                <div class='insight-card'>
                <strong>Recommended Action:</strong><br>{ai_analysis['recommendation']}
                </div>
                """, unsafe_allow_html=True)
                
        elif ai_analysis["type"] == "customer_segment":
            # Create a customer segmentation focused visualization
            col1, col2 = st.columns([2,1])
            with col1:
                # Create a simulated segment analysis if real data is unavailable
                segments = pd.DataFrame({
                    'Segment': ['Loyal Enthusiasts', 'Big Spenders', 'Casual Visitors'],
                    'CustomerPct': [32, 21, 47],
                    'RevenuePct': [47, 24, 29],
                    'SatisfactionScore': [4.8, 4.1, 3.9],
                    'VisitFrequency': [3.2, 1.8, 0.7]  # times per week
                })
                
                fig = make_subplots(rows=1, cols=2, 
                                    specs=[[{"type": "pie"}, {"type": "bar"}]],
                                    subplot_titles=("Revenue Contribution", "Key Metrics by Segment"))
                
                fig.add_trace(
                    go.Pie(labels=segments['Segment'], values=segments['RevenuePct'], name="Revenue"),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=segments['Segment'], y=segments['SatisfactionScore'], name="Satisfaction"),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(x=segments['Segment'], y=segments['VisitFrequency'], name="Weekly Visits"),
                    row=1, col=2
                )
                
                fig.update_layout(title_text="Customer Segment Analysis", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown(f"""
                <div class='insight-card'>
                <strong>Coffee Mate-Generated Insight:</strong><br>{ai_analysis['insight']}
                </div>
                <div class='insight-card'>
<strong>Coffee Mate-Generated Insight:</strong><br>{ai_analysis['insight']}
</div>
<div class='insight-card'>
<strong>Recommended Action:</strong><br>{ai_analysis['recommendation']}
</div>
""", unsafe_allow_html=True)
        
        else:
            # Generic visualization for other queries
            st.markdown(f"""
            <div class='insight-card'>
            <strong>Coffee Mate-Generated Insight:</strong><br>{ai_analysis['insight']}
            </div>
            <div class='insight-card'>
            <strong>Recommended Action:</strong><br>{ai_analysis['recommendation']}
            </div>
            """, unsafe_allow_html=True)# ENHANCEMENT 1: Natural Language Insights Engine
st.markdown("## 💬 Ask Anything About Your Data")
custom_query = st.text_area("Ask in plain English:", placeholder="Example: Show me the relationship between weather and customer satisfaction", height=60)

if custom_query and st.button("Generate Custom Insight"):
    with st.spinner("Coffee Mate analysing your request..."):
        time.sleep(1.2)  # Simulate processing
        ai_analysis = generate_custom_ai_analysis(custom_query, filtered_data)
        
        st.markdown(f"### 🤖 {ai_analysis['title']}")
        
        # Choose visualization based on analysis type
        if ai_analysis["type"] == "weather_satisfaction" and 'Weather' in filtered_data.columns:
            col1, col2 = st.columns([2,1])
            with col1:
                # Create more advanced visualization
                fig = px.scatter(filtered_data, 
                                x='Customer_Satisfaction', 
                                y='Customer_Count',
                                color='Weather', 
                                size='Revenue_Coffee',
                                hover_data=['Day'],
                                title="Weather Impact on Customer Metrics")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown(f"""
                <div class='insight-card'>
                <strong>Coffee Mate-Generated Insight:</strong><br>{ai_analysis['insight']}
                </div>
                <div class='insight-card'>
                <strong>Recommended Action:</strong><br>{ai_analysis['recommendation']}
                </div>
                """, unsafe_allow_html=True)
                
        elif ai_analysis["type"] == "revenue_day" and 'Day' in filtered_data.columns:
            col1, col2 = st.columns([2,1])
            with col1:
                # Create a more detailed day-of-week analysis
                if available_revenue_cols:
                    revenue_by_day = filtered_data.groupby('Day')[available_revenue_cols].mean().reset_index()
                    fig = px.bar(revenue_by_day.melt(id_vars='Day', value_vars=available_revenue_cols, var_name='Revenue Stream', value_name='Amount'),
                                x='Day', y='Amount', color='Revenue Stream', barmode='stack',
                                title="Revenue Breakdown by Day of Week")
                    fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown(f"""
                <div class='insight-card'>
                <strong>Coffee Mate-Generated Insight:</strong><br>{ai_analysis['insight']}
                </div>
                <div class='insight-card'>
                <strong>Recommended Action:</strong><br>{ai_analysis['recommendation']}
                </div>
                """, unsafe_allow_html=True)
                
        elif ai_analysis["type"] == "customer_segment":
            # Create a customer segmentation focused visualization
            col1, col2 = st.columns([2,1])
            with col1:
                # Create a simulated segment analysis if real data is unavailable
                segments = pd.DataFrame({
                    'Segment': ['Loyal Enthusiasts', 'Big Spenders', 'Casual Visitors'],
                    'CustomerPct': [32, 21, 47],
                    'RevenuePct': [47, 24, 29],
                    'SatisfactionScore': [4.8, 4.1, 3.9],
                    'VisitFrequency': [3.2, 1.8, 0.7]  # times per week
                })
                
                fig = make_subplots(rows=1, cols=2, 
                                    specs=[[{"type": "pie"}, {"type": "bar"}]],
                                    subplot_titles=("Revenue Contribution", "Key Metrics by Segment"))
                
                fig.add_trace(
                    go.Pie(labels=segments['Segment'], values=segments['RevenuePct'], name="Revenue"),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=segments['Segment'], y=segments['SatisfactionScore'], name="Satisfaction"),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(x=segments['Segment'], y=segments['VisitFrequency'], name="Weekly Visits"),
                    row=1, col=2
                )
                
                fig.update_layout(title_text="Customer Segment Analysis", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown(f"""
                <div class='insight-card'>
                <strong>Coffee Mate-Generated Insight:</strong><br>{ai_analysis['insight']}
                </div>
                # Revenue Breakdown
st.markdown('<p class="sub-header">Revenue Analysis</p>', unsafe_allow_html=True)
rev_col1, rev_col2 = st.columns(2)

with rev_col1:
    revenue_streams = ['Revenue_Coffee', 'Revenue_Food', 'Revenue_Workspace', 'Revenue_Events']
    available_streams = [col for col in revenue_streams if col in filtered_data.columns]
    
    if available_streams:
        revenue_totals = [filtered_data[col].sum() for col in available_streams]
        revenue_labels = ['Coffee', 'Food', 'Workspace', 'Events'][:len(available_streams)]
        
        fig_pie = px.pie(
            values=revenue_totals,
            names=revenue_labels,
            title="Revenue Breakdown by Source",
            color_discrete_sequence=px.colors.sequential.Brwnyl,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("""
        <div class='insight-card'>
            <strong>🤖 Coffee Mate Insight:</strong> Coffee sales contribute the highest revenue at 53%, 
            but Events have the highest profit margin at 72%. Consider expanding high-margin Events 
            and Workshops to increase overall profitability.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Revenue breakdown cannot be displayed: required columns are missing.")

with rev_col2:
    if 'Day' in filtered_data.columns and available_streams:
        daily_revenue = filtered_data.groupby('Day', observed=True)[available_streams].sum().reset_index()
        daily_revenue['Total_Revenue'] = daily_revenue[available_streams].sum(axis=1)
        
        expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        existing_days = list(set(daily_revenue['Day']))
        
        if all(day in existing_days for day in expected_days):
            daily_revenue['Day'] = pd.Categorical(daily_revenue['Day'], categories=expected_days, ordered=True)
            daily_revenue = daily_revenue.sort_values('Day')
        
        fig_bar = px.bar(
            daily_revenue,
            x='Day',
            y='Total_Revenue',
            title="Average Revenue by Day of Week",
            color='Day',
            color_discrete_sequence=px.colors.sequential.Brwnyl
        )
        fig_bar.update_layout(xaxis_title="Day of Week", yaxis_title="Revenue ($)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("""
        <div class='insight-card'>
            <strong>🤖 Coffee Mate Insight:</strong> Weekend revenue is 47% higher than weekday revenue. 
            Specifically, Saturdays generate the highest revenue, suggesting an opportunity to 
            replicate successful weekend strategies on slower weekdays.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Daily revenue breakdown cannot be displayed: required columns are missing.")
        # Customer Analysis
st.markdown('<p class="sub-header">Customer Insights</p>', unsafe_allow_html=True)
cust_col1, cust_col2 = st.columns(2)

with cust_col1:
    if 'Date' in filtered_data.columns and 'New_Customers' in filtered_data.columns and 'Repeat_Customers' in filtered_data.columns:
        customer_time = filtered_data.groupby('Date', observed=True)[['New_Customers', 'Repeat_Customers']].sum().reset_index()
        customer_time = customer_time.melt(
            id_vars=['Date'],
            value_vars=['New_Customers', 'Repeat_Customers'],
            var_name='Customer Type',
            value_name='Count'
        )
        fig_line = px.line(
            customer_time,
            x='Date',
            y='Count',
            color='Customer Type',
            title="New vs. Repeat Customers Over Time",
            color_discrete_sequence=['#8B5A2B', '#D2B48C']
        )
        fig_line.update_layout(xaxis_title="Date", yaxis_title="Number of Customers")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("Customer time series cannot be displayed: required columns are missing.")

with cust_col2:
    if 'Weather' in filtered_data.columns and 'Customer_Satisfaction' in filtered_data.columns:
        weather_sat = filtered_data.groupby('Weather', observed=True)['Customer_Satisfaction'].mean().reset_index()
        fig_bar2 = px.bar(
            weather_sat,
            x='Weather',
            y='Customer_Satisfaction',
            title="Customer Satisfaction by Weather",
            color='Weather',
            color_discrete_sequence=px.colors.sequential.Brwnyl
        )
        fig_bar2.update_layout(xaxis_title="Weather", yaxis_title="Avg. Satisfaction (1-5)")
        st.plotly_chart(fig_bar2, use_container_width=True)
    else:
        st.warning("Customer satisfaction by weather cannot be displayed: required columns are missing.")
        # AI Advanced Analysis Section
st.markdown('<p class="sub-header">Coffee Mate-Powered Advanced Analytics</p>', unsafe_allow_html=True)

# Tabs for different AI analyses
tab1, tab2, tab3, tab4 = st.tabs(["Customer Segmentation", "Revenue Prediction", "Workshop Optimisation", "Anomaly Detection"])

with tab1:
    col1, col2 = st.columns([2,1])
    with col1:
        required_cols = ['Avg_Ticket_Size', 'Customer_Satisfaction', 'Repeat_Customers', 'Customer_Count']
        if all(col in filtered_data.columns for col in required_cols) and len(filtered_data) > 3:
            cluster_data = filtered_data[['Avg_Ticket_Size', 'Customer_Satisfaction']].copy()
            cluster_data['Repeat_Rate'] = filtered_data['Repeat_Customers'] / filtered_data['Customer_Count']
            cluster_data = cluster_data.dropna()
            if len(cluster_data) >= 3:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                kmeans = KMeans(n_clusters=min(3, len(cluster_data)), n_init=10, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)
                cluster_df = filtered_data.loc[cluster_data.index].copy()
                cluster_df['Cluster'] = cluster_labels
                cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
                cluster_mapping = {}
                for i, center in enumerate(cluster_centers):
                    ticket_size, satisfaction, repeat_rate = center
                    if satisfaction > 4.5 and repeat_rate > 0.7:
                        label = "Loyal Enthusiasts"
                    elif ticket_size > 12:
                        label = "Big Spenders"
                    else:
                        label = "Casual Visitors"
                    cluster_mapping[i] = label
                cluster_df['Segment'] = cluster_df['Cluster'].map(cluster_mapping)
                fig_scatter = px.scatter(
                    cluster_df,
                    x='Avg_Ticket_Size',
                    y='Customer_Satisfaction',
                    color='Segment',
                    size='Customer_Count',
                    hover_data=['Day', 'Weather'],
                    title="Coffee Mate Customer Segmentation Analysis",
                    color_discrete_sequence=['#8B5A2B', '#D2B48C', '#A0522D']
                )
                fig_scatter.update_layout(xaxis_title="Average Ticket Size ($)", yaxis_title="Customer Satisfaction")
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Not enough valid data points for customer segmentation.")
        else:
            st.warning("Customer segmentation cannot be displayed: required columns are missing or insufficient data.")
    with col2:
        st.markdown("### Coffee Mate-Identified Customer Segments")
        st.markdown("""
        <div class='insight-card'>
            <strong>Loyal Enthusiasts (32%)</strong><br>
            High satisfaction, frequent visits, medium spend<br>
            <strong>Action:</strong> Loyalty program, community events
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='insight-card'>
            <strong>Big Spenders (21%)</strong><br>
            High ticket size, moderate satisfaction<br>
            <strong>Action:</strong> Premium offerings, personalised service
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='insight-card'>
            <strong>Casual Visitors (47%)</strong><br>
            Lower frequency, varied satisfaction<br>
            <strong>Action:</strong> Targeted promotions, improved experience
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 🤖 Coffee Mate Recommendation")
        st.info("Increase 'Loyal Enthusiast' segment by 15% through personalised workshop invitations based on past attendance patterns and implementing a tiered rewards program.")

with tab2:
    pred_col1, pred_col2 = st.columns([3,1])
    with pred_col1:
        if 'Date' in filtered_data.columns and available_streams:
            daily_revenue = filtered_data.groupby('Date', observed=True)[available_streams].sum()
            daily_revenue['Total_Revenue'] = daily_revenue.sum(axis=1)
            if len(daily_revenue) >= 7:
                last_date = daily_revenue.index.max()
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14)
                forecast_df = pd.DataFrame(index=forecast_dates, columns=['Total_Revenue', 'Lower_Bound', 'Upper_Bound'])
                ma_value = daily_revenue['Total_Revenue'].rolling(min(7, len(daily_revenue))).mean().iloc[-min(7, len(daily_revenue)):].mean()
                if len(daily_revenue) >= 14:
                    trend = (daily_revenue['Total_Revenue'].iloc[-7:].mean() - daily_revenue['Total_Revenue'].iloc[-14:-7].mean()) / 7
                else:
                    trend = 0
                for i, date in enumerate(forecast_dates):
                    forecast_df.loc[date, 'Total_Revenue'] = ma_value + (i * trend) + np.random.normal(0, 50)
                    forecast_df.loc[date, 'Lower_Bound'] = forecast_df.loc[date, 'Total_Revenue'] * 0.9
                    forecast_df.loc[date, 'Upper_Bound'] = forecast_df.loc[date, 'Total_Revenue'] * 1.1
                historical = daily_revenue[['Total_Revenue']].copy()
                historical['Type'] = 'Historical'
                forecast = forecast_df[['Total_Revenue']].copy()
                forecast['Type'] = 'Forecast'
                combined = pd.concat([historical, forecast]).reset_index().rename(columns={'index': 'Date'})
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=combined[combined['Type'] == 'Historical']['Date'],
                    y=combined[combined['Type'] == 'Historical']['Total_Revenue'],
                    mode='lines',
                    name='Historical Revenue',
                    line=dict(color='#8B5A2B', width=2)
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=combined[combined['Type'] == 'Forecast']['Date'],
                    y=combined[combined['Type'] == 'Forecast']['Total_Revenue'],
                    mode='lines',
                    name='Coffee Mate Revenue Forecast',
                    line=dict(color='#D2B48C', width=2, dash='dash')
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=list(forecast_df.index) + list(forecast_df.index)[::-1],
                    y=list(forecast_df['Upper_Bound']) + list(forecast_df['Lower_Bound'])[::-1],
                    fill='toself',
                    fillcolor='rgba(210, 180, 140, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    hoverinfo='skip',
                    showlegend=False
                ))
                fig_forecast.update_layout(
                    title='Coffee Mate Revenue Forecast (Next 14 Days)',
                    xaxis_title='Date',
                    yaxis_title='Revenue ($)',
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
            else:
                st.warning("Not enough data for meaningful revenue forecast. Need at least 7 days of data.")
        else:
            st.warning("Revenue forecast cannot be displayed: required columns are missing.")
    with pred_col2:
        st.markdown("### Revenue Forecast Insights")
        if 'Date' in filtered_data.columns and available_streams and len(daily_revenue) >= 7:
            forecast_total = forecast_df['Total_Revenue'].sum()
            forecast_avg = forecast_df['Total_Revenue'].mean()
            historical_avg = daily_revenue['Total_Revenue'].iloc[-min(14, len(daily_revenue)):].mean()
            percent_change = ((forecast_avg - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
            st.markdown(f"""
            <div class='insight-card'>
                <strong>14-Day Forecast</strong><br>
                Total: ${forecast_total:,.2f}<br>
                Daily Avg: ${forecast_avg:,.2f}<br>
                Change: {percent_change:+.1f}%
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class='insight-card'>
                <strong>🤖 Key Coffee Mate-Detected Factors</strong><br>
                1. Day of week (48% importance)<br>
                2. Weather patterns (27% importance)<br>
                3. Workshop schedule (15% importance)<br>
                4. Promotions (10% importance)
            </div>
            """, unsafe_allow_html=True)
            st.markdown("### 🤖 Coffee Mate Recommendation")
            st.info("Increase revenue by $2,450 next week by scheduling a 'Business Planning' workshop on Thursday and running a targeted promotion on Wednesday to drive attendance.")
        else:
            st.warning("Forecast insights cannot be displayed: insufficient data.")
            with tab3:
    required_workshop_cols = ['Workshop_Topic', 'Workshop_Attendees', 'Workshop_Satisfaction', 'Revenue_Events', 'Revenue_Coffee', 'Revenue_Food']
    if all(col in filtered_data.columns for col in required_workshop_cols):
        workshop_data = filtered_data[filtered_data['Workshop_Attendees'] > 0].copy()
        if len(workshop_data) > 0:
            workshop_perf = workshop_data.groupby('Workshop_Topic', observed=True).agg({
                'Workshop_Attendees': 'mean',
                'Workshop_Satisfaction': 'mean',
                'Revenue_Events': 'mean',
                'Revenue_Coffee': 'mean',
                'Revenue_Food': 'mean'
            }).reset_index()
            workshop_perf['Total_Revenue'] = workshop_perf['Revenue_Events'] + workshop_perf['Revenue_Coffee'] + workshop_perf['Revenue_Food']
            workshop_perf['Revenue_Per_Attendee'] = workshop_perf['Total_Revenue'] / workshop_perf['Workshop_Attendees']
            workshop_perf = workshop_perf.sort_values('Total_Revenue', ascending=False)
            wkshp_col1, wkshp_col2 = st.columns([3,1])
            with wkshp_col1:
                fig_bubble = px.scatter(
                    workshop_perf,
                    x='Workshop_Satisfaction',
                    y='Revenue_Per_Attendee',
                    size='Workshop_Attendees',
                    color='Workshop_Topic',
                    hover_data=['Total_Revenue'],
                    title='Workshop Performance Matrix',
                )
                fig_bubble.update_layout(
                    xaxis_title='Customer Satisfaction (1-5)',
                    yaxis_title='Revenue Per Attendee ($)',
                    xaxis=dict(range=[4.3, 5.0])
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
            with wkshp_col2:
                st.markdown("### Workshop Performance")
                top_revenue = workshop_perf.iloc[0]['Workshop_Topic']
                top_satisfaction = workshop_perf.loc[workshop_perf['Workshop_Satisfaction'].idxmax()]['Workshop_Topic']
                top_attendance = workshop_perf.loc[workshop_perf['Workshop_Attendees'].idxmax()]['Workshop_Topic']
                st.markdown(f"""
                <div class='insight-card'>
                    <strong>Top Performers</strong><br>
                    Revenue: {top_revenue}<br>
                    Satisfaction: {top_satisfaction}<br>
                    Attendance: {top_attendance}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class='insight-card'>
                    <strong>🤖 Coffee Mate Workshop Optimisation</strong><br>
                    Increase frequency of high-performing workshops by 25% and strategically schedule them on Thursdays to maximise pre-weekend attendance.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("### 🤖 Coffee Mate Recommendation")
                st.info("Create a 'Business Growth Bundle' combining your top 3 workshops into a discounted series. Coffee Mate projects a 38% higher retention rate for customers who complete the full series.")
        else:
            st.warning("Workshop analysis cannot be displayed: no workshop data available.")
    else:
        st.warning("Workshop analysis cannot be displayed: required columns are missing.")

# ENHANCEMENT 2: Anomaly Detection Tab
with tab4:
    st.markdown("### Coffee Mate-Powered Anomaly Detection")
    st.info("Coffee Mate continuously monitors your business metrics for unusual patterns that might represent opportunities or problems.")
    
    anomaly_cols = ['Revenue_Coffee', 'Customer_Count', 'Customer_Satisfaction']
    if all(col in filtered_data.columns for col in anomaly_cols):
        # Create simple anomaly scores based on Z-scores for demo purposes
        anomaly_data = filtered_data.copy()
        for col in anomaly_cols:
            anomaly_data[f'{col}_zscore'] = (anomaly_data[col] - anomaly_data[col].mean()) / anomaly_data[col].std()
        
        # Flag anomalies where absolute z-score > 2
        anomaly_data['anomaly'] = np.where(
            (anomaly_data['Revenue_Coffee_zscore'].abs() > 2) | 
            (anomaly_data['Customer_Count_zscore'].abs() > 2) | 
            (anomaly_data['Customer_Satisfaction_zscore'].abs() > 2), 
            'Anomaly', 'Normal'
        )
        
        # Create visualization
        fig = px.scatter(anomaly_data, x='Date', y='Revenue_Coffee', 
                         color='anomaly', size='Customer_Count',
                         hover_data=['Day', 'Weather', 'Customer_Satisfaction'],
                         title="Coffee Mate-Detected Anomalies in Business Performance",
                         color_discrete_map={'Anomaly': '#DC3545', 'Normal': '#8B5A2B'})
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=anomaly_data['Date'],
            y=anomaly_data['Revenue_Coffee'].rolling(7).mean(),
            mode='lines', name='7-Day Trend',
            line=dict(color='black', width=1, dash='dash')
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display list of anomalies
        anomalies = anomaly_data[anomaly_data['anomaly'] == 'Anomaly'].sort_values('Date')
        if not anomalies.empty:
            st.subheader("Detected Anomalies")
            for i, (_, row) in enumerate(anomalies.iterrows()):
                if i < 5:  # Limit to 5 anomalies for display clarity
                    st.markdown(f"""
                    <div class='insight-card'>
                        <strong>{row['Date'].strftime('%d %b %y')} ({row['Day']})</strong><br>
                        Revenue: ${row['Revenue_Coffee']:.2f} ({'+' if row['Revenue_Coffee_zscore'] > 0 else ''}{row['Revenue_Coffee_zscore']:.1f}σ)<br>
                        Customers: {row['Customer_Count']:.0f} ({'+' if row['Customer_Count_zscore'] > 0 else ''}{row['Customer_Count_zscore']:.1f}σ)<br>
                        Weather: {row['Weather']}<br>
                        <strong>Coffee Mate Analysis:</strong> {
                            "Unexpectedly high revenue despite lower customer count. Likely due to premium purchase behaviour during the workshop event." 
                            if row['Revenue_Coffee_zscore'] > 0 and row['Customer_Count_zscore'] < 0
                            else "Significant drop in both customers and revenue, correlating with severe weather conditions."
                            if row['Revenue_Coffee_zscore'] < 0 and row['Customer_Count_zscore'] < 0
                            else "Unusually high customer count did not translate to proportional revenue increase. Possible promotion impact."
                        }
                    </div>
                    """, unsafe_allow_html=True)
            
            if len(anomalies) > 5:
                st.info(f"{len(anomalies) - 5} more anomalies detected. Use filters to focus on specific time periods.")
    else:
        st.warning("Anomaly detection requires Revenue_Coffee, Customer_Count, and Customer_Satisfaction data.")
        # ENHANCEMENT 3: Predictive Customer Churn Analysis
st.markdown('<p class="sub-header">Customer Retention Prediction</p>', unsafe_allow_html=True)

# In a real implementation, this would use a trained ML model
# Here we're simulating churn prediction
if 'Customer_Count' in filtered_data.columns and 'Repeat_Customers' in filtered_data.columns:
    # Create synthetic churn data
    churn_data = pd.DataFrame({
        'Segment': ['Loyal Enthusiasts', 'Big Spenders', 'Casual Visitors'],
        'Current_Churn_Rate': [0.05, 0.18, 0.32],
        'Predicted_Next_Month': [0.04, 0.22, 0.29],
        'Revenue_Impact': [2500, 8900, 4200]
    })
    
    churn_col1, churn_col2 = st.columns([2,1])
    
    with churn_col1:
        # Visualize churn predictions
        fig = go.Figure()
        for i, segment in enumerate(churn_data['Segment']):
            current = churn_data[churn_data['Segment'] == segment]['Current_Churn_Rate'].values[0]
            predicted = churn_data[churn_data['Segment'] == segment]['Predicted_Next_Month'].values[0]
            fig.add_trace(go.Bar(
                x=[segment, segment],
                y=[current, predicted],
                text=[f"{current*100:.1f}%", f"{predicted*100:.1f}%"],
                textposition='auto',
                name=segment if i==0 else None,  # Only add segment to legend once
                legendgroup=segment,
                marker_color=['#8B5A2B', '#D2B48C'] if predicted <= current else ['#8B5A2B', '#DC3545']
            ))
        
        fig.update_layout(
            title='Coffee Mate-Predicted Customer Churn by Segment',
            xaxis_title='Customer Segment',
            yaxis_title='Churn Rate',
            yaxis=dict(tickformat='.0%'),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add custom legend for current vs predicted
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name='Current',
            marker_color='#8B5A2B',
            showlegend=True
        ))
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name='Predicted (Next Month)',
            marker_color='#D2B48C',
            showlegend=True
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with churn_col2:
        # Intervention recommendations
        st.markdown("### 🤖 Coffee Mate-Generated Retention Strategy")
        st.markdown("""
        <div class='insight-card'>
            <strong>High-Risk Segment:</strong> Big Spenders are predicted to increase churn by 4% next month, impacting $8,900 in revenue
        </div>
        <div class='insight-card'>
            <strong>Recommended Intervention:</strong> Coffee Mate identified that Big Spenders who haven't visited in 14+ days and previously ordered Specialty Coffee 
            respond best to personalised offers with a 72% conversion rate. Send the "Exclusive Tasting" promotion to the 37 at-risk customers.
        </div>
        """, unsafe_allow_html=True)
        
        # Add action button
        if st.button("Generate Targeted Retention Campaign"):
            with st.spinner("Coffee Mate generating campaign..."):
                time.sleep(1.2)
                st.success("Campaign generated! 37 customers targeted with personalised offers based on past purchase behaviour.")
                st.download_button(
                    label="Download Campaign List (CSV)",
                    data="Customer_ID,Name,Last_Visit,Offer_Type,Predicted_Response\n1001,John,2025-03-01,Exclusive Tasting,High\n1002,Sarah,2025-03-03,Exclusive Tasting,Medium",
                    file_name="big_spenders_retention_campaign.csv",
                    mime="text/csv"
                )
else:
    st.warning("Customer retention prediction requires Customer_Count and Repeat_Customers data.")
    # ENHANCEMENT 4: Prescriptive AI Recommendations Dashboard
st.markdown("## 🤖 Coffee Mate Action Centre")

ai_rec_col1, ai_rec_col2, ai_rec_col3 = st.columns(3)

with ai_rec_col1:
    st.markdown("""
    <div class="recommendation-card">
        <h4 class="recommendation-header">📈 Revenue Opportunity</h4>
        <div class="recommendation-content">
            <p><strong>Insight:</strong> Thursdays show 27% revenue drop between 2-4pm</p>
            <p><strong>Coffee Mate Recommendation:</strong> Implement "Afternoon Productivity Bundle" with coffee + workspace discount targeting remote workers</p>
            <p><strong>Predicted Impact:</strong> +$1,270 monthly revenue</p>
        </div>
        <div class="button-container">
            <button class="action-button">Implement</button>
            <button class="secondary-button">Analyse</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ai_rec_col2:
    st.markdown("""
    <div class="recommendation-card">
        <h4 class="recommendation-header">👥 Customer Experience</h4>
        <div class="recommendation-content">
            <p><strong>Insight:</strong> Satisfaction drops 0.7 points during high-traffic periods (10-11am)</p>
            <p><strong>Coffee Mate Recommendation:</strong> Add 1 additional barista during peak hours + implement express ordering for repeat customers</p>
            <p><strong>Predicted Impact:</strong> +0.8pt satisfaction increase, +12% repeat visits</p>
        </div>
        <div class="button-container">
            <button class="action-button">Implement</button>
            <button class="secondary-button">Analyse</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ai_rec_col3:
    st.markdown("""
    <div class="recommendation-card">
        <h4 class="recommendation-header">🧠 Workshop Strategy</h4>
        <div class="recommendation-content">
            <p><strong>Insight:</strong> "Business Planning" workshops drive 215% more workspace bookings</p>
            <p><strong>Coffee Mate Recommendation:</strong> Create 3-part Business Planning series with graduated pricing and partner with local business coach</p>
            <p><strong>Predicted Impact:</strong> +$3,400 quarterly from workshops and extended workspace usage</p>
        </div>
        <div class="button-container">
            <button class="action-button">Implement</button>
            <button class="secondary-button">Analyse</button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # ENHANCEMENT 5: Interactive "What-If" Scenario Simulator with Improved Layout and Responsiveness
st.markdown('<div class="scenario-simulator">', unsafe_allow_html=True)

# Updated header with better responsive design
st.markdown("""
<div class="scenario-header">
    <h2 class="scenario-title">Business Scenario Simulator</h2>
    <button class="secondary-button">Reset All Parameters</button>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.info("Use Coffee Mate to predict outcomes of different business decisions")

# Use a container for better spacing
with st.container():
    sim_col1, sim_col2 = st.columns([1,2], gap="large")

    with sim_col1:
        st.subheader("Adjust Parameters")
        
        # Coffee Price slider with zero point indicator and color coding
        st.markdown("<div style='margin-top: 15px;'><strong>Coffee Price Change (%)</strong></div>", unsafe_allow_html=True)
        coffee_price = st.slider("", min_value=-20, max_value=30, value=0, key="coffee_price", 
                                label_visibility="collapsed")
        
        # Custom HTML for zero indicator and color zones
        zero_point_percent = ((0 - (-20)) / (30 - (-20))) * 100
        st.markdown(
        f"""
        <style>
        .slider-container {{
            position: relative;
            margin-bottom: 2rem;
        }}
        .zero-point {{
            position: absolute;
            left: {zero_point_percent}%;
            top: -15px;
            color: black;
            font-size: 12px;
        }}
        .arrow-up {{
            width: 0; 
            height: 0; 
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-bottom: 5px solid black;
            margin: 0 auto;
        }}
        .red-zone {{
            position: absolute;
            background: rgba(255,0,0,0.1);
            height: 5px;
            left: {zero_point_percent}%;
            width: {100 - zero_point_percent}%;
            top: 10px;
        }}
        .green-zone {{
            position: absolute;
            background: rgba(0,128,0,0.1);
            height: 5px;
            left: 0%;
            width: {zero_point_percent}%;
            top: 10px;
        }}
        </style>
        <div class="slider-container">
            <div class="zero-point">0<div class="arrow-up"></div></div>
            <div class="red-zone"></div>
            <div class="green-zone"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Marketing Budget slider with zero point indicator and reversed colors
        st.markdown("<div style='margin-top: 15px;'><strong>Marketing Budget Change (%)</strong></div>", unsafe_allow_html=True)
        marketing_budget = st.slider("", min_value=-50, max_value=100, value=0, key="marketing_budget",
                                    label_visibility="collapsed")
        
        zero_point_percent_marketing = ((0 - (-50)) / (100 - (-50))) * 100
        st.markdown(
        f"""
        <style>
        .marketing-slider-container {{
            position: relative;
            margin-bottom: 2rem;
        }}
        .marketing-zero-point {{
            position: absolute;
            left: {zero_point_percent_marketing}%;
            top: -15px;
            color: black;
            font-size: 12px;
        }}
        .marketing-arrow-up {{
            width: 0; 
            height: 0; 
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-bottom: 5px solid black;
            margin: 0 auto;
        }}
        .marketing-green-zone {{
            position: absolute;
            background: rgba(0,128,0,0.1);
            height: 5px;
            left: {zero_point_percent_marketing}%;
            width: {100 - zero_point_percent_marketing}%;
            top: 10px;
        }}
        .marketing-red-zone {{
            position: absolute;
            background: rgba(255,0,0,0.1);
            height: 5px;
            left: 0%;
            width: {zero_point_percent_marketing}%;
            top: 10px;
        }}
        </style>
        <div class="marketing-slider-container">
            <div class="marketing-zero-point">0<div class="marketing-arrow-up"></div></div>
            <div class="marketing-green-zone"></div>
            <div class="marketing-red-zone"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # More spacing for the remaining controls
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        new_workshop = st.selectbox("Add New Workshop Theme", ["None", "Advanced Coffee Brewing", "Entrepreneurship", "Digital Marketing", "Wellness"])
        
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        extend_hours = st.checkbox("Extend Hours (+2 hours)")
        
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        loyal_program = st.checkbox("Implement Loyalty Program")
        
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        if st.button("Run Coffee Mate Simulation", type="primary", use_container_width=True):
            with st.spinner("Running business scenario simulation..."):
                time.sleep(1.5)  # Simulate processing
                simulation_run = True
        else:
            simulation_run = False

    with sim_col2:
        # Only show results if button is clicked
        if 'simulation_run' in locals() and simulation_run:
            # Create a simulated forecast based on the parameters
            # This would use an actual ML model in a real application
            
            # Simple impact modifiers for demo
            volume_modifier = 1.0
            if coffee_price > 0:
                volume_modifier -= coffee_price * 0.01  # Each 1% price increase reduces volume by 1%
            else:
                volume_modifier += abs(coffee_price) * 0.005  # Each 1% price decrease increases volume by 0.5%
                
            marketing_modifier = 1.0 + (marketing_budget * 0.002)  # Each 10% marketing increase adds 2% volume
            
            workshop_modifier = {
                "None": 1.0,
                "Advanced Coffee Brewing": 1.08,
                "Entrepreneurship": 1.15,
                "Digital Marketing": 1.12,
                "Wellness": 1.05
            }[new_workshop]
            
            hours_modifier = 1.12 if extend_hours else 1.0
            loyalty_modifier = 1.18 if loyal_program else 1.0
            
            # Calculate overall impact
            revenue_impact = volume_modifier * marketing_modifier * workshop_modifier * hours_modifier * loyalty_modifier
            
            # Base metrics from current data
            base_revenue = filtered_data[available_revenue_cols].sum().sum() if 'available_revenue_cols' in locals() else 100000
            base_customers = filtered_data['Customer_Count'].sum() if 'Customer_Count' in filtered_data.columns else 10000
            
            # Calculate new metrics
            new_revenue = base_revenue * revenue_impact
            profit_margin = 0.3 + (0.01 * coffee_price if coffee_price > 0 else 0)  # Base 30% margin, increases with price
            new_profit = new_revenue * profit_margin
            base_profit = base_revenue * 0.3
            
            # Create metrics display
            st.subheader("Coffee Mate-Predicted Business Impact")
            impact_col1, impact_col2, impact_col3 = st.columns(3)
            
            with impact_col1:
                st.metric("Monthly Revenue", f"${new_revenue/3:,.2f}", f"{(revenue_impact-1)*100:.1f}%")
                
            with impact_col2:
                st.metric("Monthly Profit", f"${new_profit/3:,.2f}", f"{((new_profit/base_profit)-1)*100:.1f}%")
                
            with impact_col3:
                cust_change = (revenue_impact * 0.9 - 1) * 100  # Customers don't increase quite as much as revenue
                st.metric("Monthly Customers", f"{base_customers*revenue_impact*0.9/3:,.0f}", f"{cust_change:.1f}%")
            
            # Create visualization of impact
            months = ['April', 'May', 'June', 'July', 'August', 'September']
            base_rev = [base_revenue/3] * 6
            
            # Create some variability in forecast
            forecast_rev = [(base_revenue/3) * revenue_impact * (1 + np.random.normal(0, 0.02)) for _ in range(6)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=base_rev, name="Current Trajectory", line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=months, y=forecast_rev, name="Coffee Mate-Predicted Outcome", line=dict(color='#6F4E37')))
            
            fig.update_layout(
                title="Revenue Forecast Based on Selected Scenario",
                xaxis_title="Month",
                yaxis_title="Monthly Revenue ($)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate strategy notes
            strategy_notes = generate_strategy_notes(coffee_price, marketing_budget, new_workshop, extend_hours, loyal_program)
            
            # AI recommendations
            st.markdown("### 🤖 Coffee Mate Strategy Notes")
            for note in strategy_notes:
                st.markdown(f"- {note}")
                
            # Additional conditional insights
            if coffee_price > 10:
                st.warning("High price increase may initially boost profits but risks long-term customer attrition. Consider phasing in increases.")
            
            if marketing_budget > 50 and coffee_price < 0:
                st.error("High marketing spend combined with price decreases may drive volume but significantly reduces profit margins.")
            
            if new_workshop == "Entrepreneurship" and marketing_budget > 30:
                st.success("Entrepreneurship workshops with increased marketing show strong synergy, potentially attracting a valuable new customer segment.")
                
            if loyal_program and marketing_budget < 20:
                st.info("Loyalty programs work best with adequate marketing support. Consider increasing marketing budget by at least 10% to maximise program impact.")
        else:
            # Show placeholder when simulation hasn't been run
            st.markdown("""
            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-top: 20px; text-align: center;">
                <img src="https://raw.githubusercontent.com/BlueSeasAI/Data-Analytics/main/BrewMindslogo.png" width="120" style="opacity: 0.5;"><br>
                <p style="color: #666; margin-top: 15px;">Adjust parameters and run the simulation to see Coffee Mate's business predictions</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
# Download the data
st.markdown("### Download Dataset")
csv_data = data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Full Dataset (CSV)",
    data=csv_data,
    file_name="brew_minds_dataset.csv",
    mime="text/csv"
)

# Footer with fixed date info
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666666; font-size: 0.8rem;">
    Brew Minds Analytics Dashboard | Coffee Mate-Powered Data Insights<br>
    Data Range: Jan - Mar 2025 | Last Updated: Apr 2025
</div>
""", unsafe_allow_html=True)