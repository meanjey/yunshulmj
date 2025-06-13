import os
from flask import Flask, request, render_template, send_file, url_for, send_from_directory
import pandas as pd
from ga_solver import solve_vrp
from plot import plot_results, plot_cost_breakdown
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import io
import base64
import json
from datetime import datetime
import uuid # 导入uuid模块

app = Flask(__name__, static_folder='static')

# 确保必要的目录存在
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('static/figures', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return '没有文件被上传', 400
        
        file = request.files['file']
        if file.filename == '':
            return '没有选择文件', 400
        
        if not file.filename.endswith('.csv'):
            return '请上传CSV文件', 400

        # 保存上传的文件
        file_path = os.path.join('data', 'input.csv')
        file.save(file_path)

        # 读取CSV文件
        try:
            df = pd.read_csv(file_path)
            required_columns = ['from', 'to', 'cost', 'distance', 'carbon_factor']
            if not all(col in df.columns for col in required_columns):
                return 'CSV文件格式不正确，需要包含：from, to, cost, distance, carbon_factor列', 400
        except Exception as e:
            return f'读取CSV文件时出错：{str(e)}', 400

        # 获取参数
        carbon_price = float(request.form.get('carbon_price', 50))
        time_cost = float(request.form.get('time_cost', 60))

        # 运行优化算法
        try:
            result = solve_vrp(
                df=df,
                carbon_price=carbon_price,
                time_cost=time_cost
            )

            # 准备保存的结果数据
            output_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'parameters': {
                    'carbon_price': carbon_price,
                    'time_cost': time_cost
                },
                'optimal_path': result.path,
                'costs': {
                    'freight': result.costs.freight,
                    'time': result.costs.time,
                    'carbon': result.costs.carbon,
                    'total': result.costs.total
                }
            }

            # 保存结果到JSON文件
            output_path = os.path.join('results', 'output.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)

            # 生成并保存成本饼图
            plt.clf()
            fig_pie = plot_results(result)
            figure_path_pie = os.path.join('static', 'figures', 'cost_comparison.png')
            fig_pie.savefig(figure_path_pie, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig_pie)

            # 生成并保存成本构成柱状图
            plt.clf()
            fig_bar = plot_cost_breakdown(result)
            figure_path_bar = os.path.join('static', 'figures', 'cost_breakdown.png')
            fig_bar.savefig(figure_path_bar, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig_bar)
            
            # 生成一个唯一的ID，防止浏览器缓存旧图片
            cache_buster = uuid.uuid4().hex

            return render_template('index.html', 
                                result=result,
                                fig_url=url_for('static', filename='figures/cost_comparison.png', v=cache_buster),
                                breakdown_fig_url=url_for('static', filename='figures/cost_breakdown.png', v=cache_buster))
        except Exception as e:
            return f'运行优化算法时出错：{str(e)}', 500

    return render_template('index.html')

@app.route('/download_json')
def download_json():
    """提供 results/output.json 文件的下载"""
    try:
        # 使用 send_from_directory 更安全，能防止路径遍历攻击
        return send_from_directory('results', 'output.json', as_attachment=True)
    except FileNotFoundError:
        return "JSON文件未找到，请先运行一次优化。", 404

if __name__ == '__main__':
    app.run(debug=True) 