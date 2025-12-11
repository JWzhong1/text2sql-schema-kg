import os
import json
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from pathlib import Path

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 防止中文乱码(如果有)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_latest_report(directory, prefix):
    """获取目录下以 prefix 开头的最新 JSON 文件"""
    files = glob.glob(os.path.join(directory, f"{prefix}*.json"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def plot_to_base64(fig):
    """将 matplotlib 图片转换为 base64 字符串以嵌入 HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def process_comparison_data(ours_data, dinsql_data):
    """处理两个模型的详细对比数据"""
    rows = []
    
    # 获取所有问题ID的并集
    all_ids = set(ours_data['mismatches'].keys()) | set(dinsql_data['mismatches'].keys())
    
    for q_id in all_ids:
        # --- 处理 GraphRAG (Ours) ---
        if q_id in ours_data['mismatches']:
            our_item = ours_data['mismatches'][q_id]
            o_metrics = our_item.get('metrics', {})
            o_diffs = our_item.get('differences', {})
            
            # 计算 Ours F1 (Table & Column)
            ot_tp, ot_fp, ot_fn = o_metrics.get('tbl_tp', 0), o_metrics.get('tbl_fp', 0), o_metrics.get('tbl_fn', 0)
            ot_prec = ot_tp / (ot_tp + ot_fp) if (ot_tp + ot_fp) > 0 else 0
            ot_rec = ot_tp / (ot_tp + ot_fn) if (ot_tp + ot_fn) > 0 else 0
            ot_f1 = 2 * (ot_prec * ot_rec) / (ot_prec + ot_rec) if (ot_prec + ot_rec) > 0 else 0
            
            oc_tp, oc_fp, oc_fn = o_metrics.get('col_tp', 0), o_metrics.get('col_fp', 0), o_metrics.get('col_fn', 0)
            oc_prec = oc_tp / (oc_tp + oc_fp) if (oc_tp + oc_fp) > 0 else 0
            oc_rec = oc_tp / (oc_tp + oc_fn) if (oc_tp + oc_fn) > 0 else 0
            oc_f1 = 2 * (oc_prec * oc_rec) / (oc_prec + oc_rec) if (oc_prec + oc_rec) > 0 else 0
            
            o_missing_col = len(o_diffs.get('missing_columns', []))
            o_extra_col = len(o_diffs.get('extra_columns', []))
            question_text = our_item.get('question', '')
        else:
            # 如果不在 mismatches 中，说明完全正确 (F1 = 1.0)
            ot_f1, oc_f1 = 1.0, 1.0
            o_missing_col, o_extra_col = 0, 0
            question_text = ""

        # --- 处理 DIN-SQL ---
        if q_id in dinsql_data['mismatches']:
            din_item = dinsql_data['mismatches'][q_id]
            d_metrics = din_item.get('metrics', {})
            d_diffs = din_item.get('differences', {})
            
            # 计算 DIN F1 (Table & Column)
            dt_tp, dt_fp, dt_fn = d_metrics.get('tbl_tp', 0), d_metrics.get('tbl_fp', 0), d_metrics.get('tbl_fn', 0)
            dt_prec = dt_tp / (dt_tp + dt_fp) if (dt_tp + dt_fp) > 0 else 0
            dt_rec = dt_tp / (dt_tp + dt_fn) if (dt_tp + dt_fn) > 0 else 0
            dt_f1 = 2 * (dt_prec * dt_rec) / (dt_prec + dt_rec) if (dt_prec + dt_rec) > 0 else 0
            
            dc_tp, dc_fp, dc_fn = d_metrics.get('col_tp', 0), d_metrics.get('col_fp', 0), d_metrics.get('col_fn', 0)
            dc_prec = dc_tp / (dc_tp + dc_fp) if (dc_tp + dc_fp) > 0 else 0
            dc_rec = dc_tp / (dc_tp + dc_fn) if (dc_tp + dc_fn) > 0 else 0
            dc_f1 = 2 * (dc_prec * dc_rec) / (dc_prec + dc_rec) if (dc_prec + dc_rec) > 0 else 0
            
            d_missing_col = len(d_diffs.get('missing_columns', []))
            d_extra_col = len(d_diffs.get('extra_columns', []))
            
            # 如果 question_text 还没获取到（即 Ours 是完全正确的），则从 DIN 数据中获取
            if not question_text:
                question_text = din_item.get('question', '')
        else:
            # 如果不在 mismatches 中，说明完全正确 (F1 = 1.0)
            dt_f1, dc_f1 = 1.0, 1.0
            d_missing_col, d_extra_col = 0, 0

        row = {
            'Question ID': q_id,
            'Question': question_text,
            
            'Ours Table F1': ot_f1, 'Ours Col F1': oc_f1,
            'DIN Table F1': dt_f1, 'DIN Col F1': dc_f1,
            
            'Table F1 Diff': ot_f1 - dt_f1,
            'Col F1 Diff': oc_f1 - dc_f1,
            
            'Ours Missing Col': o_missing_col,
            'DIN Missing Col': d_missing_col,
            'Ours Extra Col': o_extra_col,
            'DIN Extra Col': d_extra_col
        }
        rows.append(row)
        
    return pd.DataFrame(rows)

def generate_charts(ours_summary, dinsql_summary, df):
    charts = {}
    
    # 1. 总体指标对比 (Grouped Bar Chart)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    metrics = ['Precision', 'Recall', 'F1']
    levels = ['Table', 'Column']
    
    plot_data = []
    for level in levels:
        key = f"{level.lower()}_metrics"
        for m in metrics:
            plot_data.append({'Model': 'GraphRAG (Ours)', 'Metric': f"{level} {m}", 'Value': ours_summary[key][m.lower()]})
            plot_data.append({'Model': 'DIN-SQL', 'Metric': f"{level} {m}", 'Value': dinsql_summary[key][m.lower()]})
            
    sns.barplot(data=pd.DataFrame(plot_data), x='Metric', y='Value', hue='Model', ax=ax1, palette=['#4e79a7', '#f28e2b'])
    ax1.set_title('Overall Performance Comparison', fontsize=14)
    ax1.set_ylim(0, 1.15)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
    charts['overall'] = plot_to_base64(fig1)

    # 2. F1 分数分布对比 (KDE Plot)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x='Ours Col F1', fill=True, label='GraphRAG (Ours)', color='#4e79a7', alpha=0.3, ax=ax2)
    sns.kdeplot(data=df, x='DIN Col F1', fill=True, label='DIN-SQL', color='#f28e2b', alpha=0.3, ax=ax2)
    ax2.set_title('Distribution of Column F1 Scores', fontsize=14)
    ax2.set_xlabel('F1 Score')
    ax2.set_xlim(0, 1)
    ax2.legend()
    charts['f1_dist'] = plot_to_base64(fig2)

    # 3. Win/Loss 分析 (Histogram of Differences)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    # 过滤掉 0 差异的，只看有变化的
    diffs = df[df['Col F1 Diff'] != 0]['Col F1 Diff']
    colors = ['#e15759' if x < 0 else '#59a14f' for x in diffs]
    
    if len(diffs) > 0:
        sns.histplot(x=diffs, bins=20, ax=ax3, hue=diffs > 0, palette={True: '#59a14f', False: '#e15759'}, legend=False)
        ax3.set_title(f'Performance Delta (Ours - DIN) per Question\n(Green: Ours Better, Red: DIN Better)', fontsize=14)
        ax3.set_xlabel('F1 Score Difference')
        ax3.axvline(0, color='black', linestyle='--', linewidth=1)
    else:
        ax3.text(0.5, 0.5, "No Performance Difference Found", ha='center', va='center')
        
    charts['win_loss'] = plot_to_base64(fig3)

    # 4. 错误类型统计 (Stacked Bar)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    error_data = [
        {'Model': 'GraphRAG', 'Type': 'Missing Col', 'Count': df['Ours Missing Col'].sum()},
        {'Model': 'DIN-SQL', 'Type': 'Missing Col', 'Count': df['DIN Missing Col'].sum()},
        {'Model': 'GraphRAG', 'Type': 'Extra Col', 'Count': df['Ours Extra Col'].sum()},
        {'Model': 'DIN-SQL', 'Type': 'Extra Col', 'Count': df['DIN Extra Col'].sum()},
    ]
    sns.barplot(data=pd.DataFrame(error_data), x='Type', y='Count', hue='Model', ax=ax4, palette=['#4e79a7', '#f28e2b'])
    ax4.set_title('Total Schema Linking Errors Comparison', fontsize=14)
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%d')
    charts['errors'] = plot_to_base64(fig4)

    return charts

def generate_html_report(db_name, ours_summary, dinsql_summary, df, charts, output_path):
    
    # 找出提升最大和下降最大的 Top 5
    improved = df.sort_values(by='Col F1 Diff', ascending=False).head(5)
    regressed = df.sort_values(by='Col F1 Diff', ascending=True).head(5)
    
    # 仅保留有差异的
    improved = improved[improved['Col F1 Diff'] > 0]
    regressed = regressed[regressed['Col F1 Diff'] < 0]

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comparison Report: {db_name}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f4f6f9; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #4e79a7; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 40px; border-left: 5px solid #f28e2b; padding-left: 10px; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
            .card {{ background: #fff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; text-align: center; }}
            .card h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #7f8c8d; }}
            .card .val {{ font-size: 20px; font-weight: bold; }}
            .card .diff {{ font-size: 12px; margin-top: 5px; }}
            .pos {{ color: #27ae60; }} .neg {{ color: #c0392b; }}
            .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
            .chart-box {{ background: white; border: 1px solid #eee; padding: 10px; border-radius: 8px; }}
            img {{ width: 100%; height: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
            th {{ background: #f8f9fa; padding: 12px; text-align: left; border-bottom: 2px solid #ddd; }}
            td {{ padding: 10px; border-bottom: 1px solid #eee; }}
            tr:hover {{ background: #f1f1f1; }}
            .badge {{ padding: 3px 8px; border-radius: 10px; font-size: 11px; color: white; }}
            .b-blue {{ background: #4e79a7; }} .b-orange {{ background: #f28e2b; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Comparison: GraphRAG vs DIN-SQL ({db_name})</h1>
            <p>Generated Comparison Report. <span class="badge b-blue">GraphRAG (Ours)</span> vs <span class="badge b-orange">DIN-SQL (Baseline)</span></p>
            
            <div class="summary-grid">
                <div class="card">
                    <h3>Table F1</h3>
                    <div class="val">{ours_summary['table_metrics']['f1']:.4f}</div>
                    <div class="diff {( 'pos' if ours_summary['table_metrics']['f1'] >= dinsql_summary['table_metrics']['f1'] else 'neg' )}">
                        vs {dinsql_summary['table_metrics']['f1']:.4f}
                    </div>
                </div>
                <div class="card">
                    <h3>Column F1</h3>
                    <div class="val">{ours_summary['column_metrics']['f1']:.4f}</div>
                    <div class="diff {( 'pos' if ours_summary['column_metrics']['f1'] >= dinsql_summary['column_metrics']['f1'] else 'neg' )}">
                        vs {dinsql_summary['column_metrics']['f1']:.4f}
                    </div>
                </div>
                <div class="card">
                    <h3>Avg Missing Cols</h3>
                    <div class="val">{df['Ours Missing Col'].mean():.2f}</div>
                    <div class="diff">vs {df['DIN Missing Col'].mean():.2f} (DIN)</div>
                </div>
                <div class="card">
                    <h3>Win Rate (Col F1)</h3>
                    <div class="val">{len(df[df['Col F1 Diff'] > 0])} / {len(df)}</div>
                    <div class="diff">Questions Improved</div>
                </div>
            </div>

            <h2>Visual Analysis</h2>
            <div class="chart-grid">
                <div class="chart-box"><img src="data:image/png;base64,{charts['overall']}"></div>
                <div class="chart-box"><img src="data:image/png;base64,{charts['win_loss']}"></div>
                <div class="chart-box"><img src="data:image/png;base64,{charts['f1_dist']}"></div>
                <div class="chart-box"><img src="data:image/png;base64,{charts['errors']}"></div>
            </div>

            <h2>Top Improvements (Ours > DIN)</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width:50px">ID</th>
                        <th>Question</th>
                        <th style="width:80px">Ours F1</th>
                        <th style="width:80px">DIN F1</th>
                        <th style="width:80px">Gain</th>
                    </tr>
                </thead>
                <tbody>
    """
    for _, row in improved.iterrows():
        html += f"""
            <tr>
                <td>{row['Question ID']}</td>
                <td>{row['Question']}</td>
                <td><b>{row['Ours Col F1']:.2f}</b></td>
                <td>{row['DIN Col F1']:.2f}</td>
                <td class="pos">+{row['Col F1 Diff']:.2f}</td>
            </tr>
        """
    
    html += """
                </tbody>
            </table>

            <h2>Top Regressions (DIN > Ours)</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width:50px">ID</th>
                        <th>Question</th>
                        <th style="width:80px">Ours F1</th>
                        <th style="width:80px">DIN F1</th>
                        <th style="width:80px">Loss</th>
                    </tr>
                </thead>
                <tbody>
    """
    for _, row in regressed.iterrows():
        html += f"""
            <tr>
                <td>{row['Question ID']}</td>
                <td>{row['Question']}</td>
                <td>{row['Ours Col F1']:.2f}</td>
                <td><b>{row['DIN Col F1']:.2f}</b></td>
                <td class="neg">{row['Col F1 Diff']:.2f}</td>
            </tr>
        """

    html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Comparison HTML report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize comparison between GraphRAG and DIN-SQL results")
    parser.add_argument("--db_name", type=str, required=True, help="Database name (e.g., california_schools)")
    args = parser.parse_args()

    db_name = args.db_name
    
    # 定义路径
    project_root = Path(__file__).parent.parent.parent
    result_base_dir = project_root / "scripts/evaluate/result_2025_12_10" / db_name
    dinsql_base_dir = project_root / "scripts/evaluate/dinsql_result" / db_name
    output_dir = project_root / "scripts/evaluate/comparison_charts"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载 GraphRAG (Ours) 结果
    ours_file = get_latest_report(result_base_dir, f"eval_report_{db_name}")
    
    # 2. 加载 DIN-SQL 结果
    dinsql_file = os.path.join(dinsql_base_dir, f"dinsql_retrieval_report_{db_name}.json")

    if not ours_file or not os.path.exists(ours_file):
        print(f"Error: Could not find evaluation report in {result_base_dir}")
        return

    if not os.path.exists(dinsql_file):
        print(f"Error: Could not find DIN-SQL report at {dinsql_file}")
        return

    print(f"Loading Ours: {ours_file}")
    print(f"Loading DIN-SQL: {dinsql_file}")

    ours_data = load_json(ours_file)
    dinsql_data = load_json(dinsql_file)

    # 处理数据
    df = process_comparison_data(ours_data, dinsql_data)
    
    # 生成图表
    charts = generate_charts(ours_data['summary'], dinsql_data['summary'], df)
    
    # 生成 HTML 报告
    output_html = os.path.join(output_dir, f'comparison_report_{db_name}.html')
    generate_html_report(db_name, ours_data['summary'], dinsql_data['summary'], df, charts, output_html)

if __name__ == "__main__":
    main()