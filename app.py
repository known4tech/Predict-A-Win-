import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Declaring the teams
teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

# declaring the venues
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))

st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide"
)

st.title('🏏 IPL Win Predictor')
st.markdown("Predict the probability of a team winning in T20 cricket matches")

# Input validation helper function
def validate_inputs(batting_team, bowling_team, target, score, overs, wickets):
    errors = []
    
    if batting_team == bowling_team:
        errors.append("Batting and bowling teams cannot be the same")
    
    if target < 0 or target > 300:
        errors.append("Target should be between 0 and 300")
    
    if score < 0:
        errors.append("Score cannot be negative")
    
    if overs < 0 or overs > 20:
        errors.append("Overs should be between 0 and 20")
    
    if wickets < 0 or wickets > 10:
        errors.append("Wickets should be between 0 and 10")
    
    if score > target:
        errors.append("Current score cannot exceed target")
    
    return errors

# Team selection
col1, col2 = st.columns(2)
with col1:
    battingteam = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

city = st.selectbox(
    'Select the city where the match is being played', sorted(cities))

# Target input with validation
target = int(st.number_input(
    'Target', 
    min_value=0, 
    max_value=300, 
    value=150, 
    step=1,
    help="Target score for the batting team (0-300)"
))

# Match situation inputs with proper validation
col3, col4, col5 = st.columns(3)
with col3:
    score = int(st.number_input(
        'Score', 
        min_value=0, 
        value=0, 
        step=1,
        help="Current score of the batting team (must be ≥ 0)"
    ))
    
with col4:
    overs = st.number_input(
        'Overs Completed', 
        min_value=0.0, 
        max_value=20.0, 
        value=0.0, 
        step=0.1,
        help="Overs completed (0.0 to 20.0)"
    )
    
with col5:
    wickets = int(st.number_input(
        'Wickets Fallen', 
        min_value=0, 
        max_value=10, 
        value=0, 
        step=1,
        help="Number of wickets fallen (0-10)"
    ))

# Display current match status
if score > 0 and overs > 0:
    current_rr = score / overs
    st.info(f"Current Run Rate: {current_rr:.2f}")

# Validate inputs
validation_errors = validate_inputs(battingteam, bowlingteam, target, score, overs, wickets)

if validation_errors:
    for error in validation_errors:
        st.error(error)
elif score > target:
    st.success(f"🎉 {battingteam} won the match!")
elif score == target-1 and overs == 20:
    st.warning("⚖️ Match Drawn")
elif wickets == 10 and score < target-1:
    st.success(f"🎉 {bowlingteam} won the match!")
else:
    if st.button('🎯 Predict Win Probability', type="primary"):
        try:
            if overs == 0:
                st.error("Cannot calculate probability at 0 overs completed")
            else:
                runs_left = target - score
                balls_left = 120 - (overs * 6)
                wickets_remaining = 10 - wickets
                currentrunrate = score / overs
                requiredrunrate = (runs_left * 6) / balls_left if balls_left > 0 else 0
                
                input_df = pd.DataFrame({
                    'batting_team': [battingteam], 
                    'bowling_team': [bowlingteam], 
                    'city': [city], 
                    'runs_left': [runs_left], 
                    'balls_left': [balls_left], 
                    'wickets': [wickets_remaining], 
                    'total_runs_x': [target], 
                    'cur_run_rate': [currentrunrate], 
                    'req_run_rate': [requiredrunrate]
                })
                
                result = pipe.predict_proba(input_df)
                loss_prob = result[0][0]
                win_prob = result[0][1]
                
                # Display results with improved UI
                st.markdown("## 📊 Win Probability")
                
                # Create probability chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=[battingteam, bowlingteam],
                        y=[win_prob * 100, loss_prob * 100],
                        marker_color=['#1f77b4', '#ff7f0e'],
                        text=[f'{win_prob*100:.1f}%', f'{loss_prob*100:.1f}%'],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Win Probability Comparison",
                    xaxis_title="Teams",
                    yaxis_title="Win Probability (%)",
                    yaxis=dict(range=[0, 100]),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display match statistics
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    st.metric(
                        label=f"{battingteam} Win Probability",
                        value=f"{win_prob*100:.1f}%"
                    )
                    st.metric(
                        label="Required Run Rate",
                        value=f"{requiredrunrate:.2f}"
                    )
                
                with col_stats2:
                    st.metric(
                        label=f"{bowlingteam} Win Probability",
                        value=f"{loss_prob*100:.1f}%"
                    )
                    st.metric(
                        label="Current Run Rate",
                        value=f"{currentrunrate:.2f}"
                    )
                
                # Additional match info
                st.markdown("### 📈 Match Statistics")
                stats_data = {
                    'Metric': ['Runs Required', 'Balls Remaining', 'Wickets in Hand', 'Run Rate Difference'],
                    'Value': [runs_left, balls_left, wickets_remaining, f"{requiredrunrate - currentrunrate:.2f}"]
                }
                st.dataframe(pd.DataFrame(stats_data), hide_index=True)
                
        except ZeroDivisionError:
            st.error("⚠️ Please fill all the required details correctly")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
