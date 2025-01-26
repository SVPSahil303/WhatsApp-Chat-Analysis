from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from plotly import graph_objects as go
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import emoji
import os
from io import BytesIO

app = Flask(__name__)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Helper Functions
def preprocess(data):
    pattern = '\\d{1,2}/\\d{1,2}/\\d{2,4},\\s\\d{1,2}:\\d{2}\\s-\\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)
    users, messages = [], []
    for message in df['user_message']:
        entry = re.split('([\\w\\W]+?):\\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)
    return pd.DataFrame(df)

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud.to_file('static/wordcloud.png')

def analyze_sentiment(messages):
    return messages['message'].apply(lambda msg: sia.polarity_scores(msg)['compound'] if pd.notnull(msg) else 0)

def extract_emojis(s):
    """Extract emojis from a string."""
    return [char for char in s if char in emoji.EMOJI_DATA]

    return emojis

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    chat_data = preprocess(data)

    if chat_data.empty:
        return jsonify({'error': 'No valid data found in the chat file'}), 400

    # Save chat data to CSV for report download
    csv_path = "uploads/chat_data.csv"
    chat_data.to_csv(csv_path, index=False)

    # Perform analysis
    user_stats = chat_data['user'].value_counts().to_dict()
    most_active_user = chat_data['user'].value_counts().idxmax()
    all_messages = ' '.join(chat_data['message'].dropna())
    generate_wordcloud(all_messages)
    sentiments = analyze_sentiment(chat_data)
    chat_data['sentiment'] = sentiments
    # Extract all emojis
    all_emojis = ''.join(chat_data['message'].dropna().apply(lambda x: ''.join(extract_emojis(x))))

    # Count and sort emojis
    emoji_counts = Counter(all_emojis)
    sorted_emoji_counts = dict(sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True))
    messages_by_hour = chat_data.groupby(chat_data['date'].dt.hour).size().to_dict()
    top_words = Counter(" ".join(chat_data['message']).split()).most_common(10)

    # Visualization: User Activity Over Time
    plt.figure(figsize=(10, 6))
    for user in chat_data['user'].unique():
        user_data = chat_data[chat_data['user'] == user]
        user_data.set_index('date').resample('D').size().plot(label=user)
    plt.title("User Activity Over Time")
    plt.xlabel("Date")
    plt.ylabel("Messages Sent")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('static/user_activity_over_time.png')
    plt.close()

    # Visualization: Day-wise Activity Heatmap
    chat_data['day_of_week'] = chat_data['date'].dt.day_name()
    heatmap_data = chat_data.groupby(['day_of_week', chat_data['date'].dt.hour]).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=False)
    plt.title("Day-wise Activity Heatmap")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig('static/day_wise_activity_heatmap.png')
    plt.close()

    # Visualization: Network Graph (using Plotly)
    # Visualization: Network Graph (using Plotly)
    edges = []
    for _, row in chat_data.iterrows():
        if row['user'] != 'group_notification':
            # Extract mentioned users (if applicable)
            mentioned_users = [word.strip('@') for word in row['message'].split() if '@' in word]
            for mentioned_user in mentioned_users:
                edges.append((row['user'], mentioned_user))

    # If no edges are found, handle the error gracefully
    if not edges:
        return jsonify({'error': 'No mentions found in the chat messages to create a network graph.'}), 400

    # Create graph
    G = nx.DiGraph(edges)
    pos = nx.spring_layout(G, seed=42)  # Generate layout positions for nodes

    # Prepare data for Plotly
    node_x = []
    node_y = []
    node_names = []
    for node, position in pos.items():
        node_x.append(position[0])
        node_y.append(position[1])
        node_names.append(node)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create edge traces
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_names,
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            size=10,
            color='lightblue',
            line_width=2
        )
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Network Graph of Interactions',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    # Save as HTML
    fig.write_html('static/network_graph.html')

    return render_template('results.html',
                           user_stats=user_stats,
                           most_active_user=most_active_user,
                           messages_by_hour=messages_by_hour,
                           emoji_counts=sorted_emoji_counts,
                           wordcloud_path='static/wordcloud.png',
                           top_words=top_words,
                           user_activity_graph='static/user_activity_over_time.png',
                           heatmap='static/day_wise_activity_heatmap.png',
                           network_graph='static/network_graph.html')


@app.route('/download', methods=['GET'])
def download_report():
    try:
        # Read the saved CSV file from the uploads directory
        csv_path = os.path.join("uploads", "chat_data.csv")

        if not os.path.exists(csv_path):
            return jsonify({'error': 'Report not generated yet. Please upload a file first.'}), 400

        # Open the CSV file and encode it as bytes
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_data = f.read()

        # Convert string data to a BytesIO object for download
        output = BytesIO()
        output.write(csv_data.encode('utf-8'))  # Encode as bytes
        output.seek(0)

        # Send file as attachment
        return send_file(output, mimetype="text/csv", as_attachment=True, download_name="chat_analysis_report.csv")

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
