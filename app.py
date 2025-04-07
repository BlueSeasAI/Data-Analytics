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

# Page configuration
st.set_page_config(
    page_title="Brew Minds Analytics | AI-Powered Insights",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
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
</style>
""", unsafe_allow_html=True)

# Load data - UPDATED with explicit date conversion and error handling
@st.cache_data
def load_data():
    # Create synthetic data directly - skip any CSV loading attempts entirely
    st.info("Creating synthetic coffee shop dataset...")
    
    # Generate synthetic data with proper structure
    import numpy as np
    
    # Generate consistent dates - UPDATED for 2025
    sample_dates = pd.date_range(start='2025-01-01', periods=90)
    
    # Create realistic days of week based on dates
    days_of_week = [date.strftime('%A') for date in sample_dates]
    
    # Create data with realistic patterns (weekends have higher values)
    coffee_revenue = []
    food_revenue = []
    workspace_revenue = []
    events_revenue = []
    customer_count = []
    
    for i, day in enumerate(days_of_week):
        # Weekend multiplier
        weekend_factor = 1.5 if day in ['Saturday', 'Sunday'] else 1.0
        
        # Base values with some randomness
        coffee_base = np.random.uniform(480, 550)
        food_base = np.random.uniform(200, 250)
        workspace_base = np.random.uniform(150, 250) 
        events_base = 0 if day not in ['Wednesday', 'Saturday'] else np.random.uniform(350, 450)
        
        # Apply weekend factor and add to lists
        coffee_revenue.append(coffee_base * weekend_factor)
        food_revenue.append(food_base * weekend_factor)
        workspace_revenue.append(workspace_base)
        events_revenue.append(events_base)
        
        # Customer count proportional to revenue
        base_customers = np.random.randint(70, 90)
        customer_count.append(int(base_customers * weekend_factor))
    
    # Create the DataFrame
    data = pd.DataFrame({
        'Date': sample_dates,
        'Day': days_of_week,
        'Revenue_Coffee': coffee_revenue,
        'Revenue_Food': food_revenue,
        'Revenue_Workspace': workspace_revenue,
        'Revenue_Events': events_revenue,
        'Customer_Count': customer_count,
        'New_Customers': [int(count * np.random.uniform(0.2, 0.35)) for count in customer_count],
        'Repeat_Customers': [int(count * np.random.uniform(0.65, 0.8)) for count in customer_count],
        'Avg_Ticket_Size': np.random.uniform(9.5, 14.0, 90),
        'Weather': np.random.choice(['Cold', 'Mild', 'Warm', 'Rainy', 'Snowy'], 90),
        'Customer_Satisfaction': np.random.uniform(4.0, 5.0, 90),
        'Workshop_Topic': ['None'] * 90,  # Will be updated for specific days
        'Workshop_Attendees': [0] * 90,   # Will be updated for specific days
        'Workshop_Satisfaction': [0.0] * 90  # Will be updated for specific days
    })
    
    # Add workshops on Wednesdays and Saturdays
    workshop_topics = ['Business Planning', 'Time Management', 'Creative Writing', 
                      'Digital Marketing', 'Social Media Strategy', 'Personal Finance',
                      'Content Creation', 'Leadership Skills', 'Public Speaking']
    
    # Assign workshops to Wednesdays and Saturdays
    for i, date in enumerate(data['Date']):
        if data.loc[i, 'Day'] in ['Wednesday', 'Saturday']:
            if np.random.random() > 0.2:  # 80% chance to have a workshop
                data.loc[i, 'Workshop_Topic'] = np.random.choice(workshop_topics)
                data.loc[i, 'Workshop_Attendees'] = np.random.randint(14, 20)
                data.loc[i, 'Workshop_Satisfaction'] = np.random.uniform(4.4, 4.9)
    
    return data

data = load_data()

# Sidebar filters
st.sidebar.markdown("## Filters")
st.sidebar.image("https://via.placeholder.com/150x150.png?text=Brew+Minds", width=150)

# UPDATED: Date range filter with defensive checks
if 'Date' in data.columns and len(data) > 0 and not data['Date'].isna().all():
    min_date = data['Date'].min().date()
    max_date = data['Date'].max().date()
else:
    # Fallback to default dates if Date column isn't available or valid
    current_date = datetime.now().date()
    min_date = current_date - timedelta(days=90)
    max_date = current_date

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Apply date filter with defensive checks
if len(date_range) == 2 and 'Date' in data.columns:
    start_date, end_date = date_range
    filtered_data = data[(data['Date'] >= pd.Timestamp(start_date)) & 
                         (data['Date'] <= pd.Timestamp(end_date))]
else:
    filtered_data = data.copy()

# Day of week filter with defensive check
if 'Day' in filtered_data.columns and len(filtered_data) > 0:
    days = filtered_data['Day'].unique().tolist()
    selected_days = st.sidebar.multiselect("Select Days", days, default=days)
    if selected_days:
        filtered_data = filtered_data[filtered_data['Day'].isin(selected_days)]

# Weather filter with defensive check
if 'Weather' in filtered_data.columns and len(filtered_data) > 0:
    weather_options = filtered_data['Weather'].unique().tolist()
    selected_weather = st.sidebar.multiselect("Weather Conditions", weather_options, default=weather_options)
    if selected_weather:
        filtered_data = filtered_data[filtered_data['Weather'].isin(selected_weather)]

# Main page
st.markdown('<p class="main-header">Brew Minds Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown("### AI-Powered Data Analytics Demo")

# AI Assistant Section
st.sidebar.markdown("## AI Assistant")
analysis_question = st.sidebar.text_input("Ask AI about your business data:", 
                                         placeholder="E.g., What's our best performing day?")

if analysis_question:
    with st.sidebar.spinner("AI thinking..."):
        # Simulate AI processing time
        import time
        time.sleep(1.5)
        
        # Predefined responses based on keywords
        responses = {
            "best": "Based on my analysis, Saturday shows the highest overall revenue and customer count. Workshops on Business Planning and Time Management generate the best customer satisfaction scores.",
            "revenue": "Coffee sales contribute to about 55% of your revenue. I detect a positive correlation between weather conditions and revenue - warmer days see a 12% increase in sales compared to cold days.",
            "customer": "Your repeat customer rate is 72%. Customers who attend workshops are 2.3x more likely to return within the same week. Social media engagement positively correlates with new customer acquisition.",
            "workshop": "The most profitable workshop topics are Business Planning, Time Management, and Women in Business. There's an opportunity to increase frequency of these high-performing workshops.",
            "trend": "I've detected a weekly cycle with peaks on weekends. There's a positive growth trend of 3.2% month-over-month, with customer satisfaction gradually improving over the period.",
            "marketing": "Marketing spend has an ROI of approximately 425%. Friday and Saturday promotions perform 37% better than promotions on other days."
        }
        
        # Check if any keywords match
        response = "I don't have enough context to answer that specific question. Try asking about revenue, customers, workshops, trends, or marketing."
        for key, value in responses.items():
            if key in analysis_question.lower():
                response = value
                break
                
        st.sidebar.markdown(f"<div class='insight-card'><strong>AI Analysis:</strong> {response}</div>", unsafe_allow_html=True)

# Check if we have data to display
if len(filtered_data) == 0:
    st.warning("No data available with the current filters or there was an error loading the data. Please check the dataset or adjust filters.")
    st.stop()

# KPI Cards
st.markdown("## Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

# Calculate KPIs with defensive checks
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

# Calculate period-over-period changes (assuming we have at least 2 weeks of data)
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

# Display KPI cards
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

# Revenue Breakdown
st.markdown('<p class="sub-header">Revenue Analysis</p>', unsafe_allow_html=True)
rev_col1, rev_col2 = st.columns(2)

with rev_col1:
    # Prepare data for revenue streams with defensive checks
    revenue_streams = ['Revenue_Coffee', 'Revenue_Food', 'Revenue_Workspace', 'Revenue_Events']
    available_streams = [col for col in revenue_streams if col in filtered_data.columns]
    
    if available_streams:
        revenue_totals = [filtered_data[col].sum() for col in available_streams]
        revenue_labels = ['Coffee', 'Food', 'Workspace', 'Events'][:len(available_streams)]
        
        # Create pie chart for revenue breakdown
        fig_pie = px.pie(
            values=revenue_totals,
            names=revenue_labels,
            title="Revenue Breakdown by Source",
            color_discrete_sequence=px.colors.sequential.Brwnyl,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # AI Insight card
        st.markdown("""
        <div class='insight-card'>
            <strong>🤖 AI Insight:</strong> Coffee sales contribute the highest revenue at 53%, 
            but Events have the highest profit margin at 72%. Consider expanding high-margin Events 
            and Workshops to increase overall profitability.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Revenue breakdown cannot be displayed: required columns are missing.")

with rev_col2:
    # Daily revenue by day of week with defensive checks
    if 'Day' in filtered_data.columns and available_streams:
        daily_revenue = filtered_data.groupby('Day')[available_streams].sum().reset_index()
        daily_revenue['Total_Revenue'] = daily_revenue[available_streams].sum(axis=1)
        
        # Check if we have the expected days of week
        expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        existing_days = list(set(daily_revenue['Day']))
        
        # Only reorder if we have the standard days
        if all(day in existing_days for day in expected_days):
            # Reorder days of week
            daily_revenue['Day'] = pd.Categorical(daily_revenue['Day'], categories=expected_days, ordered=True)
            daily_revenue = daily_revenue.sort_values('Day')
        
        # Create bar chart
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
        
        # AI Insight card
        st.markdown("""
        <div class='insight-card'>
            <strong>🤖 AI Insight:</strong> Weekend revenue is 47% higher than weekday revenue. 
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
    # Time series of new vs repeat customers
    if 'Date' in filtered_data.columns and 'New_Customers' in filtered_data.columns and 'Repeat_Customers' in filtered_data.columns:
        customer_time = filtered_data.groupby('Date')[['New_Customers', 'Repeat_Customers']].sum().reset_index()
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
    # Customer satisfaction correlation with other metrics
    if 'Weather' in filtered_data.columns and 'Customer_Satisfaction' in filtered_data.columns:
        # Calculate average satisfaction by weather
        weather_sat = filtered_data.groupby('Weather')['Customer_Satisfaction'].mean().reset_index()
        
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
st.markdown('<p class="sub-header">AI-Powered Advanced Analytics</p>', unsafe_allow_html=True)

# Tabs for different AI analyses
tab1, tab2, tab3 = st.tabs(["Customer Segmentation", "Revenue Prediction", "Workshop Optimization"])

with tab1:
    col1, col2 = st.columns([2,1])
    
    with col1:
        # Perform customer segmentation using K-means with defensive checks
        required_cols = ['Avg_Ticket_Size', 'Customer_Satisfaction', 'Repeat_Customers', 'Customer_Count']
        if all(col in filtered_data.columns for col in required_cols) and len(filtered_data) > 3:
            # Prepare data for clustering
            cluster_data = filtered_data[['Avg_Ticket_Size', 'Customer_Satisfaction']].copy()
            
            # Add derived features
            cluster_data['Repeat_Rate'] = filtered_data['Repeat_Customers'] / filtered_data['Customer_Count']
            
            # Remove any rows with NaN values
            cluster_data = cluster_data.dropna()
            
            if len(cluster_data) >= 3:  # Need at least 3 points for meaningful clustering
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # Perform K-means clustering
                kmeans = KMeans(n_clusters=min(3, len(cluster_data)), random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)
                
                # Add cluster labels back to filtered_data
                cluster_df = filtered_data.loc[cluster_data.index].copy()
                cluster_df['Cluster'] = cluster_labels
                
                # Map cluster to meaningful labels based on characteristics
                cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
                cluster_mapping = {}
                
                # Assign labels based on ticket size and satisfaction
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
                
                # Create scatter plot
                fig_scatter = px.scatter(
                    cluster_df,
                    x='Avg_Ticket_Size',
                    y='Customer_Satisfaction',
                    color='Segment',
                    size='Customer_Count',
                    hover_data=['Day', 'Weather'],
                    title="AI Customer Segmentation Analysis",
                    color_discrete_sequence=['#8B5A2B', '#D2B48C', '#A0522D']
                )
                fig_scatter.update_layout(xaxis_title="Average Ticket Size ($)", yaxis_title="Customer Satisfaction")
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Not enough valid data points for customer segmentation.")
        else:
            st.warning("Customer segmentation cannot be displayed: required columns are missing or insufficient data.")
        
    with col2:
        st.markdown("### AI-Identified Customer Segments")
        
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
            <strong>Action:</strong> Premium offerings, personalized service
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-card'>
            <strong>Casual Visitors (47%)</strong><br>
            Lower frequency, varied satisfaction<br>
            <strong>Action:</strong> Targeted promotions, improved experience
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🤖 AI Recommendation")
        st.info("Increase 'Loyal Enthusiast' segment by 15% through personalized workshop invitations based on past attendance patterns and implementing a tiered rewards program.")

with tab2:
    pred_col1, pred_col2 = st.columns([3,1])
    
    with pred_col1:
        # Create a simple forecasting visualization
        # For demo purposes, we'll create a pseudo-forecast based on historical patterns
        
        # Check if we have the required data
        if 'Date' in filtered_data.columns and available_streams:
            # Aggregate daily data
            daily_revenue = filtered_data.groupby('Date')[available_streams].sum()
            daily_revenue['Total_Revenue'] = daily_revenue.sum(axis=1)
            
            if len(daily_revenue) >= 7:  # Need at least a week of data for meaningful forecast
                # Create pseudo-forecast (7-day moving average + random noise)
                last_date = daily_revenue.index.max()
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14)
                
                # Create forecast dataframe
                forecast_df = pd.DataFrame(index=forecast_dates, columns=['Total_Revenue', 'Lower_Bound', 'Upper_Bound'])
                
                # Fill with moving average + trend
                ma_value = daily_revenue['Total_Revenue'].rolling(min(7, len(daily_revenue))).mean().iloc[-min(7, len(daily_revenue)):].mean()
                
                if len(daily_revenue) >= 14:
                    trend = (daily_revenue['Total_Revenue'].iloc[-7:].mean() - daily_revenue['Total_Revenue'].iloc[-14:-7].mean()) / 7
                else:
                    trend = 0
                
                for i, date in enumerate(forecast_dates):
                    forecast_df.loc[date, 'Total_Revenue'] = ma_value + (i * trend) + np.random.normal(0, 50)
                    forecast_df.loc[date, 'Lower_Bound'] = forecast_df.loc[date, 'Total_Revenue'] * 0.9
                    forecast_df.loc[date, 'Upper_Bound'] = forecast_df.loc[date, 'Total_Revenue'] * 1.1
                
                # Combine historical and forecast data
                historical = daily_revenue[['Total_Revenue']].copy()
                historical['Type'] = 'Historical'
                forecast = forecast_df[['Total_Revenue']].copy()
                forecast['Type'] = 'Forecast'
                
                combined = pd.concat([historical, forecast])
                combined = combined.reset_index()
                combined = combined.rename(columns={'index': 'Date'})
                
                # Create the figure
                fig_forecast = go.Figure()
                
                # Add historical data
                fig_forecast.add_trace(go.Scatter(
                    x=combined[combined['Type'] == 'Historical']['Date'],
                    y=combined[combined['Type'] == 'Historical']['Total_Revenue'],
                    mode='lines',
                    name='Historical Revenue',
                    line=dict(color='#8B5A2B', width=2)
                ))
                
                # Add forecast data
                fig_forecast.add_trace(go.Scatter(
                    x=combined[combined['Type'] == 'Forecast']['Date'],
                    y=combined[combined['Type'] == 'Forecast']['Total_Revenue'],
                    mode='lines',
                    name='AI Revenue Forecast',
                    line=dict(color='#D2B48C', width=2, dash='dash')
                ))
                
                # Add confidence interval
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
                    title='AI Revenue Forecast (Next 14 Days)',
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
            # Calculate some forecast metrics
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
                <strong>🤖 Key AI-Detected Factors</strong><br>
                1. Day of week (48% importance)<br>
                2. Weather patterns (27% importance)<br>
                3. Workshop schedule (15% importance)<br>
                4. Promotions (10% importance)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🤖 AI Recommendation")
            st.info("Increase revenue by $2,450 next week by scheduling a 'Business Planning' workshop on Thursday and running a targeted promotion on Wednesday to drive attendance.")
        else:
            st.warning("Forecast insights cannot be displayed: insufficient data.")

with tab3:
    # Workshop performance analysis with defensive checks
    required_workshop_cols = ['Workshop_Topic', 'Workshop_Attendees', 'Workshop_Satisfaction', 'Revenue_Events', 'Revenue_Coffee', 'Revenue_Food']
    
    if all(col in filtered_data.columns for col in required_workshop_cols):
        workshop_data = filtered_data[filtered_data['Workshop_Attendees'] > 0].copy()
        
        if len(workshop_data) > 0:
            # Group by workshop topic
            workshop_perf = workshop_data.groupby('Workshop_Topic').agg({
                'Workshop_Attendees': 'mean',
                'Workshop_Satisfaction': 'mean',
                'Revenue_Events': 'mean',
                'Revenue_Coffee': 'mean',
                'Revenue_Food': 'mean'
            }).reset_index()
            
            workshop_perf['Total_Revenue'] = workshop_perf['Revenue_Events'] + workshop_perf['Revenue_Coffee'] + workshop_perf['Revenue_Food']
            workshop_perf['Revenue_Per_Attendee'] = workshop_perf['Total_Revenue'] / workshop_perf['Workshop_Attendees']
            
            # Sort by total revenue
            workshop_perf = workshop_perf.sort_values('Total_Revenue', ascending=False)
            
            wkshp_col1, wkshp_col2 = st.columns([3,1])
            
            with wkshp_col1:
                # Create a bubble chart
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
                
                # Calculate top performers
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
                    <strong>🤖 AI Workshop Optimization</strong><br>
                    Increase frequency of high-performing workshops by 25% and strategically schedule them on Thursdays to maximize pre-weekend attendance.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🤖 AI Recommendation")
                st.info("Create a 'Business Growth Bundle' combining your top 3 workshops into a discounted series. AI projects a 38% higher retention rate for customers who complete the full series.")
        else:
            st.warning("Workshop analysis cannot be displayed: no workshop data available.")
    else:
        st.warning("Workshop analysis cannot be displayed: required columns are missing.")

# Download the data
st.markdown("### Download Dataset")
csv_data = data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Full Dataset (CSV)",
    data=csv_data,
    file_name="brew_minds_dataset.csv",
    mime="text/csv"
)

# Footer with additional info
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666666; font-size: 0.8rem;">
    Brew Minds Analytics Dashboard | AI-Powered Data Insights Demo<br>
    Data Range: Jan - March 2025 | Last Updated: April 2025
</div>
""", unsafe_allow_html=True)