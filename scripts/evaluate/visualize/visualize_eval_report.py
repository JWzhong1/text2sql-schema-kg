import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import os
import glob

# =================配置区域=================
# 基础结果目录
BASE_RESULT_DIR = 'scripts/evaluate/dinsql_result/'
# =========================================

def load_data(filepath):
    """加载 JSON 数据"""
    if not os.path.exists(filepath):
        print(f"错误: 找不到文件 {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_mismatches(mismatches_data):
    """将 mismatches 字典转换为 Pandas DataFrame"""
    rows = []
    for q_id, item in mismatches_data.items():
        metrics = item.get('metrics', {})
        diffs = item.get('differences', {})
        
        # 计算单个问题的 Table F1
        t_tp = metrics.get('tbl_tp', 0)
        t_fp = metrics.get('tbl_fp', 0)
        t_fn = metrics.get('tbl_fn', 0)
        t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
        t_f1 = 2 * (t_prec * t_rec) / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0

        # 计算单个问题的 Column F1
        c_tp = metrics.get('col_tp', 0)
        c_fp = metrics.get('col_fp', 0)
        c_fn = metrics.get('col_fn', 0)
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0
        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
        c_f1 = 2 * (c_prec * c_rec) / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0

        row = {
            'Question ID': q_id,
            'Question': item.get('question', ''),
            'Table TP': t_tp,
            'Table FP': t_fp,
            'Table FN': t_fn,
            'Table F1': t_f1,
            'Column TP': c_tp,
            'Column FP': c_fp,
            'Column FN': c_fn,
            'Column F1': c_f1,
            'Missing Tables': len(diffs.get('missing_tables', [])),
            'Extra Tables': len(diffs.get('extra_tables', [])),
            'Missing Columns': len(diffs.get('missing_columns', [])),
            'Extra Columns': len(diffs.get('extra_columns', []))
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def plot_to_base64(fig):
    """将 matplotlib 图片转换为 base64 字符串以嵌入 HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_visualizations(summary, df):
    """生成各类图表"""
    charts = {}
    sns.set_theme(style="whitegrid")

    # 1. 总体指标概览 (Bar Chart)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1'] * 2,
        'Type': ['Table'] * 3 + ['Column'] * 3,
        'Value': [
            summary['table_metrics']['precision'], summary['table_metrics']['recall'], summary['table_metrics']['f1'],
            summary['column_metrics']['precision'], summary['column_metrics']['recall'], summary['column_metrics']['f1']
        ]
    }
    sns.barplot(data=pd.DataFrame(metrics_data), x='Metric', y='Value', hue='Type', ax=ax1, palette='viridis')
    ax1.set_title('Overall Performance Metrics (Table vs Column)', fontsize=15)
    ax1.set_ylim(0, 1.1)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.3f')
    charts['overall_metrics'] = plot_to_base64(fig1)

    # 2. 错误类型分布 (Stacked Bar)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    error_counts = df[['Missing Tables', 'Extra Tables', 'Missing Columns', 'Extra Columns']].sum()
    sns.barplot(x=error_counts.index, y=error_counts.values, ax=ax2, palette='magma')
    ax2.set_title('Total Count of Schema Errors', fontsize=15)
    ax2.set_ylabel('Count')
    ax2.bar_label(ax2.containers[0])
    charts['error_distribution'] = plot_to_base64(fig2)

    # 3. F1 分数分布 (Histogram/KDE)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x='Table F1', fill=True, label='Table F1', alpha=0.5, ax=ax3)
    sns.kdeplot(data=df, x='Column F1', fill=True, label='Column F1', alpha=0.5, ax=ax3)
    ax3.set_title('Distribution of F1 Scores per Question', fontsize=15)
    ax3.set_xlabel('F1 Score')
    ax3.legend()
    charts['f1_distribution'] = plot_to_base64(fig3)

    # 4. 混淆矩阵指标分布 (Box Plot)
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    confusion_data = df[['Table TP', 'Table FP', 'Table FN', 'Column TP', 'Column FP', 'Column FN']]
    sns.boxplot(data=confusion_data, ax=ax4, palette='Set2')
    ax4.set_title('Distribution of TP/FP/FN Counts per Question', fontsize=15)
    charts['confusion_stats'] = plot_to_base64(fig4)

    return charts

def generate_html_report(summary, df, charts, output_path):
    """生成 HTML 报告"""
    
    # 找出表现最差的 10 个问题 (基于 Column F1 升序，然后 Table F1 升序)
    worst_cases = df.sort_values(by=['Column F1', 'Table F1', 'Missing Columns']).head(10)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text2SQL Evaluation Report - {summary['db_name']}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .summary-box {{ display: flex; justify-content: space-around; background: #ecf0f1; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .metric-card {{ text-align: center; }}
            .metric-val {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; }}
            .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .chart-container {{ background: white; border: 1px solid #eee; padding: 10px; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; color: white; }}
            .bg-red {{ background-color: #e74c3c; }}
            .bg-green {{ background-color: #27ae60; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Evaluation Report: {summary['db_name']}</h1>
            <p>Generated at: {summary['generated_at']} | Total Questions: {summary['total_questions']}</p>
            
            <div class="summary-box">
                <div class="metric-card">
                    <div class="metric-val">{summary['table_metrics']['f1']:.4f}</div>
                    <div class="metric-label">Table F1</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{summary['table_metrics']['precision']:.4f}</div>
                    <div class="metric-label">Table Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{summary['table_metrics']['recall']:.4f}</div>
                    <div class="metric-label">Table Recall</div>
                </div>
                <div class="metric-card" style="border-left: 1px solid #ccc; padding-left: 40px;">
                    <div class="metric-val">{summary['column_metrics']['f1']:.4f}</div>
                    <div class="metric-label">Column F1</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{summary['column_metrics']['precision']:.4f}</div>
                    <div class="metric-label">Column Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{summary['column_metrics']['recall']:.4f}</div>
                    <div class="metric-label">Column Recall</div>
                </div>
            </div>

            <h2>Visual Analysis</h2>
            <div class="chart-grid">
                <div class="chart-container"><img src="data:image/png;base64,{charts['overall_metrics']}"></div>
                <div class="chart-container"><img src="data:image/png;base64,{charts['error_distribution']}"></div>
                <div class="chart-container"><img src="data:image/png;base64,{charts['f1_distribution']}"></div>
                <div class="chart-container"><img src="data:image/png;base64,{charts['confusion_stats']}"></div>
            </div>

            <h2>Top 10 Challenging Questions (Lowest F1 Scores)</h2>
            <p>These questions had the lowest Column F1 scores, indicating significant schema mismatches.</p>
            <table>
                <thead>
                    <tr>
                        <th style="width: 50px;">ID</th>
                        <th>Question</th>
                        <th style="width: 80px;">Table F1</th>
                        <th style="width: 80px;">Col F1</th>
                        <th>Errors (Miss/Extra)</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for _, row in worst_cases.iterrows():
        error_desc = []
        if row['Missing Tables'] > 0: error_desc.append(f"Miss Tbl: {row['Missing Tables']}")
        if row['Extra Tables'] > 0: error_desc.append(f"Extra Tbl: {row['Extra Tables']}")
        if row['Missing Columns'] > 0: error_desc.append(f"Miss Col: {row['Missing Columns']}")
        if row['Extra Columns'] > 0: error_desc.append(f"Extra Col: {row['Extra Columns']}")
        
        html_content += f"""
                    <tr>
                        <td>{row['Question ID']}</td>
                        <td>{row['Question']}</td>
                        <td>{row['Table F1']:.2f}</td>
                        <td>{row['Column F1']:.2f}</td>
                        <td><span style="color: #c0392b;">{', '.join(error_desc)}</span></td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"报告已生成: {os.path.abspath(output_path)}")

def process_single_task(task_name):
    """处理单个任务的可视化流程"""
    print(f"\n{'='*20}")
    print(f"正在处理任务: {task_name}")
    print(f"{'='*20}")
    
    result_dir = os.path.join(BASE_RESULT_DIR, task_name)
    
    # 自动获取目录下的单个 JSON 文件
    json_files = glob.glob(os.path.join(result_dir, '*.json'))
    if not json_files:
        print(f"警告: 在目录 {result_dir} 下未找到任何 JSON 文件，跳过此任务。")
        return

    json_file_path = json_files[0]
    print(f"已自动定位评估报告: {json_file_path}")

    output_html_path = f'scripts/evaluate/visualize/dinsql_evaluation_report_{task_name}.html'

    print("正在加载数据...")
    data = load_data(json_file_path)
    if not data:
        return

    print("正在处理指标...")
    df = process_mismatches(data['mismatches'])
    
    print("正在生成可视化图表...")
    charts = generate_visualizations(data['summary'], df)
    
    print("正在生成 HTML 报告...")
    generate_html_report(data['summary'], df, charts, output_html_path)

def main():
    if not os.path.exists(BASE_RESULT_DIR):
        print(f"错误: 结果目录 {BASE_RESULT_DIR} 不存在")
        return

    # 获取所有子目录作为任务列表
    tasks = [d for d in os.listdir(BASE_RESULT_DIR) if os.path.isdir(os.path.join(BASE_RESULT_DIR, d))]
   
    
    if not tasks:
        print(f"未在 {BASE_RESULT_DIR} 下发现任何任务文件夹")
        return

    print(f"发现 {len(tasks)} 个任务待处理: {tasks}")
    
    for task in tasks[:5]:
        try:
            process_single_task(task)
        except Exception as e:
            print(f"处理任务 {task} 时出错: {e}")

if __name__ == "__main__":
    main()