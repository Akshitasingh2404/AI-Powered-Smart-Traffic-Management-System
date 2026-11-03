import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

DB_PATH = "database/traffic.db"

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the SQLite database tables.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            vehicle_count INTEGER,
            car_count INTEGER,
            bike_count INTEGER,
            bus_count INTEGER,
            truck_count INTEGER,
            density REAL,
            congestion_level TEXT,
            green_time REAL,
            yellow_time REAL,
            red_time REAL,
            average_wait REAL,
            camera_id TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_historical_data(limit=100):
    """
    Retrieves latest logs from database.
    """
    init_db()
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM traffic_data ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

def generate_mock_trends():
    """
    Generates standard mock data for previewing analytics when the database is empty.
    """
    now = datetime.now()
    timestamps = [datetime(now.year, now.month, now.day, hour, 0).strftime('%Y-%m-%d %H:%M:%S') for hour in range(24)]
    
    mock_data = {
        'timestamp': timestamps,
        'vehicle_count': [12, 8, 5, 4, 15, 35, 52, 45, 38, 30, 25, 28, 35, 32, 29, 36, 58, 62, 50, 32, 22, 18, 15, 13],
        'car_count': [8, 5, 3, 2, 9, 22, 33, 28, 24, 19, 15, 17, 21, 20, 18, 22, 37, 40, 31, 20, 14, 11, 9, 8],
        'bike_count': [2, 1, 1, 1, 3, 8, 12, 10, 8, 6, 5, 6, 8, 7, 6, 8, 13, 14, 11, 7, 5, 4, 3, 3],
        'bus_count': [1, 1, 0, 0, 1, 2, 3, 3, 3, 2, 2, 2, 3, 2, 2, 3, 4, 4, 4, 2, 1, 1, 1, 1],
        'truck_count': [1, 1, 1, 1, 2, 3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 3, 2, 2, 2, 1],
        'density': [0.18, 0.12, 0.08, 0.06, 0.22, 0.48, 0.72, 0.63, 0.55, 0.43, 0.36, 0.39, 0.49, 0.45, 0.41, 0.50, 0.81, 0.86, 0.70, 0.45, 0.31, 0.25, 0.21, 0.19],
        'congestion_level': ["Light", "No Traffic", "No Traffic", "No Traffic", "Light", "Moderate", "Heavy", "Heavy", "Moderate", "Moderate", "Light", "Light", "Moderate", "Moderate", "Light", "Moderate", "Heavy", "Heavy", "Heavy", "Moderate", "Light", "Light", "Light", "Light"],
        'average_wait': [22, 20, 20, 20, 25, 42, 68, 61, 53, 41, 35, 37, 45, 41, 38, 46, 75, 82, 67, 41, 30, 25, 23, 22]
    }
    return pd.DataFrame(mock_data)

def generate_charts():
    """
    Generates JSON Plotly charts. If no db records, fall back to mock data trends.
    """
    df = get_historical_data(500)
    
    # Check if empty, fallback
    if df.empty or len(df) < 5:
        df = generate_mock_trends()
        is_preview = True
    else:
        is_preview = False
        # Convert timestamp strings to datetime objects and sort ascending for time plots
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp')
        
    charts = {}
    
    # Chart 1: Vehicle Count Trend
    fig1 = px.line(df, x='timestamp', y='vehicle_count', title='Total Vehicle Count Over Time',
                   labels={'vehicle_count': 'Vehicles', 'timestamp': 'Time'})
    fig1.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1'
    )
    charts['vehicle_count_trend'] = fig1.to_json()
    
    # Chart 2: Traffic Density Trend
    fig2 = px.area(df, x='timestamp', y='density', title='Traffic Road Density Ratio',
                   labels={'density': 'Density %', 'timestamp': 'Time'})
    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1'
    )
    charts['density_trend'] = fig2.to_json()
    
    # Chart 3: Peak Traffic Time (Average vehicles by hour)
    # Parse hour if it's datetime, else mock has hourly keys
    if not is_preview:
        df['hour'] = df['timestamp'].dt.hour
    else:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
    hourly_avg = df.groupby('hour')['vehicle_count'].mean().reset_index()
    fig3 = px.bar(hourly_avg, x='hour', y='vehicle_count', title='Average Vehicles by Hour of Day',
                  labels={'vehicle_count': 'Avg Vehicle Count', 'hour': 'Hour of Day'})
    fig3.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1'
    )
    charts['peak_traffic'] = fig3.to_json()
    
    # Chart 4: Average Waiting Time Trend
    fig4 = px.line(df, x='timestamp', y='average_wait', title='Dynamic Waiting Time (Seconds)',
                   labels={'average_wait': 'Seconds', 'timestamp': 'Time'})
    fig4.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1'
    )
    charts['waiting_time_trend'] = fig4.to_json()
    
    # Chart 5: Congestion Level Distribution
    congestion_counts = df['congestion_level'].value_counts().reset_index()
    congestion_counts.columns = ['congestion_level', 'count']
    fig5 = px.pie(congestion_counts, names='congestion_level', values='count', title='Congestion Frequency')
    fig5.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1'
    )
    charts['congestion_distribution'] = fig5.to_json()
    
    return charts, is_preview

def write_csv_report(output_filepath):
    """
    Dumps all table data to CSV.
    """
    df = get_historical_data(5000)
    if df.empty:
        df = generate_mock_trends()
    df.to_csv(output_filepath, index=False)
    return output_filepath

def write_pdf_report(output_filepath):
    """
    Compiles a professional ReportLab PDF summary report.
    """
    df = get_historical_data(100)
    if df.empty:
        df = generate_mock_trends()
        
    doc = SimpleDocTemplate(output_filepath, pagesize=letter,
                            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=15,
        alignment=1 # Centered
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=12,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#475569'),
        spaceAfter=10
    )
    
    story = []
    
    # 1. Header Section
    story.append(Paragraph("AI-Powered Smart Traffic Management System", title_style))
    story.append(Paragraph(f"<b>Traffic Summary Analytics Report</b>", ParagraphStyle('Sub', parent=title_style, fontSize=12, spaceAfter=20)))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Paragraph(f"Reporting Camera ID: CAM-001 (Main Intersection)", body_style))
    story.append(Spacer(1, 15))
    
    # 2. Key Metrics Table
    total_records = len(df)
    avg_vehicles = round(df['vehicle_count'].mean(), 1)
    peak_vehicles = int(df['vehicle_count'].max())
    avg_density = f"{round(df['density'].mean() * 100, 1)}%"
    avg_wait = f"{round(df['average_wait'].mean(), 1)}s"
    
    summary_data = [
        ["Metric Title", "Computed Value", "Analysis Notes"],
        ["Total Data Points Processed", str(total_records), "Active interval logging frequency"],
        ["Average Vehicle Throughput", str(avg_vehicles), "Cars, bikes, trucks, and buses combined"],
        ["Peak Vehicle Count", str(peak_vehicles), "Max density window detected"],
        ["Average Intersection Density", avg_density, "Average space occupancy ratio"],
        ["Average Signal Waiting Time", avg_wait, "Average wait duration per phase"]
    ]
    
    summary_table = Table(summary_data, colWidths=[180, 120, 220])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('TOPPADDING', (0,0), (-1,0), 8),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ('TOPPADDING', (0,1), (-1,-1), 6),
    ]))
    
    story.append(Paragraph("System Operational Metrics Summary", section_title_style))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # 3. Log Records
    story.append(Paragraph("Recent Log Detail Entries (Last 15 Records)", section_title_style))
    
    # Pick last 15 records
    records_to_show = df.head(15)
    
    log_data = [["Timestamp", "Count", "Cars", "Bikes", "Buses/Trucks", "Density", "Congestion", "Green Time"]]
    for idx, row in records_to_show.iterrows():
        # Parse timestamp safely
        ts = row['timestamp']
        if isinstance(ts, str) and len(ts) > 19:
            ts = ts[:19]
        elif hasattr(ts, 'strftime'):
            ts = ts.strftime('%Y-%m-%d %H:%M:%S')
            
        heavy_count = int(row['bus_count'] + row['truck_count'])
        density_label = f"{round(float(row['density'])*100, 1)}%"
        
        log_data.append([
            str(ts),
            str(row['vehicle_count']),
            str(row['car_count']),
            str(row['bike_count']),
            str(heavy_count),
            density_label,
            str(row['congestion_level']),
            f"{int(row['green_time'])}s"
        ])
        
    log_table = Table(log_data, colWidths=[120, 45, 45, 45, 75, 55, 80, 55])
    log_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
        ('TOPPADDING', (0,1), (-1,-1), 4),
    ]))
    story.append(log_table)
    
    # Build Document
    doc.build(story)
    return output_filepath
