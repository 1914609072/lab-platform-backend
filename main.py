from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
import numpy as np
import json

app = FastAPI()

# ----------------- 请求体定义 -------------------- #


# ------------------------------------------------ #

# 从数据库获取锂电池数据 -- 范晓非
@app.get("/data/lithium")
async def get_lithium_data():
    pass

# 从数据库获取燃料电池数据 -- 由佳茹
@app.get("/data/fuel")
async def get_fuel_data():
    pass

# 锂电池包数学建模 -- 高晋源
@app.get("/model/lithum")
async def run_lithium_model():
    pass

# 燃料电池系统数学建模 -- 陆晓蒙
@app.get("/model/fuel")
async def run_fuel_model():
    pass

# 车辆动力学数学建模 -- 朱展锋
@app.get("/model/car",
         summary="车辆动力模型计算",
         response_description="功率序列（W）",
         tags=["动力系统"])
async def run_car_model(
        v_sequence: str = Query(..., description="车速历史数据（单位：m/s，JSON 格式）"),
        A: Optional[float] = 2.0,
        efficiency: Optional[float] = 0.95,
        M: Optional[float] = 1500.0,
        g: Optional[float] = 9.8,
        delta: Optional[float] = 1.04,
        rolling_resistance_coefficient: Optional[float] = 0.0015,
        drag_coefficient: Optional[float] = 0.3,
        theta: Optional[float] = 0.0,
        air_density: Optional[float] = 1.202,
        a_sequence: Optional[List[float]] = None,
):
    """
    车辆动力模型计算接口

    输入车速序列和车辆参数，返回电池需求功率序列

    - **v_sequence**: 车速历史数据（单位：m/s，JSON 格式）
    - **A**: 车辆迎风面积（默认2.0m²）
    - **efficiency**: 传动效率（默认0.95）
    - **M**: 整车质量（默认1500kg）
    - **g**: 重力加速度（默认9.8m/s²）
    - **delta**: 旋转质量换算系数（默认1.04）
    - **rolling_resistance_coefficient**: 滚动阻力系数（默认0.015）
    - **drag_coefficient**: 空气阻力系数（默认0.3）
    - **theta**: 道路坡度（默认0弧度）
    - **air_density**: 空气密度（默认1.202kg/m³）
    - **a_sequence**: 加速度序列（可选，默认通过车速计算）
    """
    # 将 JSON 字符串解析为列表
    v_list = json.loads(v_sequence)

    # 数据预处理
    v_array = np.array(v_list)
    a_array = np.array(a_sequence) if a_sequence is not None else None

    # 如果未提供加速度序列，则通过车速序列计算加速度
    if a_array is None:
        a_array = np.concatenate(([0], np.diff(v_array)))  # 使用前向差分计算加速度

    # 计算各种阻力
    rolling_resistance = rolling_resistance_coefficient * M * g * np.cos(theta)  # 滚动阻力
    gradient_resistance = M * g * np.sin(theta)  # 坡道阻力
    air_resistance = 0.5 * A * air_density * drag_coefficient * v_array ** 2  # 风阻
    acceleration_resistance = delta * M * a_array  # 加速阻力

    # 计算车辆需求功率
    Pre = (rolling_resistance + gradient_resistance + air_resistance + acceleration_resistance) * v_array

    # 计算电池输出功率
    Pe = np.where(Pre > 0, Pre / efficiency, Pre * efficiency)

    # 返回结果
    return JSONResponse(
        status_code=200,
        content={
            "data": {
                "velocity": v_list,
                "power": Pe.tolist()
            }
        }
    )
    pass

# 基于遗传算法的模型参数标定方法 -- 陆晓蒙
@app.get("/algorithm/litium/calibration/genetic")
async def run_genetic_algorithm():
    pass

# 给予马尔科夫链预测模型的车辆速度预测算法 -- 刘俊
@app.get("/algorithm/car/speed_predition/markov")
async def run_markov_algorithm():
    pass

# 基于K-means聚类算法车辆工况识别算法 -- 朱展锋
@app.get("/algorithm/car/condition_recognition/kmeans")
async def run_kmens_algorithm():
    pass

# 锂电池一致性评价算法 -- 王世超
@app.get("/algorithm/lithium/battery_consistency_assessment")
async def run_battery_consistency_assessment():
    pass

# 锂电池故障检测算法 -- 洪钟申
@app.get("/algorithm/lithium/fault_detection")
async def run_fault_detection():
    pass

# 锂电池健康状态估计算法 -- 张颖
@app.get("/algorithm/lithium/soh")
async def run_soh_calculation():
    pass

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
