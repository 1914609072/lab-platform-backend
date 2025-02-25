from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
import uvicorn

app = FastAPI()

# ----------------- 请求体定义 -------------------- #
try:
    # 驾驶模式识别模型参数
    kmeans_Tr = scipy.io.loadmat('Transformation_Matrix.mat')['Tr']
    kmeans_centers = scipy.io.loadmat('Cluster_Centers.mat')['centers']
except FileNotFoundError as e:
    raise RuntimeError(f"参数加载失败: {e}")

# ------------------------------------------------ #

# 从数据库获取锂电池数据 -- 范晓非
@app.get("/data/lithium")
async def get_lithium_data():
    pass

# 从数据库获取燃料电池数据 -- 由佳茹
@app.get("/data/fuel")
async def get_fuel_data():
    pass

# 锂电池包数学建模 -- 高晋源，陆晓蒙
@app.get("/model/lithum")
async def run_lithium_model():
    pass

# 燃料电池系统数学建模 -- 刘俊，李泽宇
@app.get("/model/fuel")
async def run_fuel_model():
    pass

# 车辆动力学数学建模 -- 朱展锋
@app.get("/model/car",
         summary="车辆动力学模型",
         description="输入速度序列，返回需求功率序列")
async def run_car_model(
        v_sequence: str = Query(..., alias="v", description="速度序列(m/s)"),
        A: float = Query(2.0, alias="S", description="车辆迎风面积(默认2m²)"),
        efficiency: float = Query(0.95, alias="η", description="传动效率(默认0.95)"),
        M: float = Query(1500.0, description="车辆质量(默认1500kg)"),
        g: float = Query(9.8, description="重力加速度(默认9.8m/s²)"),
        delta: float = Query(1.04, alias="Δ", description="旋转质量换算系数(默认1.04)"),
        rolling_resistance_coefficient: float = Query(0.015, alias="μ", description="滚动阻力系数(默认0.015)"),
        drag_coefficient: float = Query(0.3, alias="C", description="空气阻力系数(默认0.3)"),
        theta: float = Query(0.0, alias="θ", description="道路坡度(默认为0)"),
        air_density: float = Query(1.202, alias="ρ", description="空气密度(默认1.202kg/m³)"),
        a_sequence: Optional[str] = Query(None, alias="a", description="加速度序列(默认通过速度序列计算)")
):
    try:
        # 解析速度序列
        v_list = [float(v) for v in v_sequence.split()]  # 将字符串 "10 20 30" 转换为列表 [10, 20, 30]

        # 解析加速度序列（如果提供）
        a_list = [float(a) for a in a_sequence.split()] if a_sequence else None

        # 处理逻辑
        v_array = np.array(v_list)
        a_array = np.array(a_list) if a_list else np.concatenate(([0], np.diff(v_array)))

        # 计算功率
        rolling_resistance = rolling_resistance_coefficient * M * g * np.cos(theta)
        gradient_resistance = M * g * np.sin(theta)
        air_resistance = 0.5 * A * air_density * drag_coefficient * v_array ** 2
        acceleration_resistance = delta * M * a_array

        Pre = (rolling_resistance + gradient_resistance + air_resistance + acceleration_resistance) * v_array
        Pe = np.where(Pre > 0, Pre / efficiency, Pre * efficiency)

        return JSONResponse(content={"data": {"velocity": v_list, "power": Pe.tolist()}})
    except Exception as e:
        raise HTTPException(400, detail=f"错误: {str(e)}")
    pass

# 基于遗传算法的模型参数标定方法 -- 陆晓蒙
@app.get("/algorithm/litium/calibration/genetic")
async def run_genetic_algorithm():
    pass

# 基于马尔科夫链预测模型的车辆速度预测算法 -- 刘俊
@app.get("/algorithm/car/speed_predition/markov")
async def run_markov_algorithm():
    pass

# 基于K-means聚类算法车辆工况识别算法 -- 朱展锋
@app.get("/algorithm/car/condition_recognition/kmeans",
         summary="驾驶模式识别模块",
         description="输入速度序列，返回驾驶模式序列")
async def run_kmeans_algorithm(
        speed: str = Query(..., alias="V",
                           description="速度采样序列（单位：m/s）"),
        window_size: int = Query(100, alias="T",
                                 description="采样窗口长度（默认100s）")
):
    try:
        # 加载输入数据并进行单位转换
        speed_array = 3.6 * np.array([float(v) for v in speed.split()])

        if len(speed_array) < window_size:
            raise HTTPException(400,
                                detail=f"速度序列长度需≥{window_size}（当前：{len(speed_array)}）")

        # 特征提取
        L = len(speed_array)
        features = np.zeros((L, 12))

        for i in range(window_size - 1, L):
            window = speed_array[i - window_size + 1: i + 1]

            # 加速度分解
            delta_v = np.diff(window)
            pos_acc = delta_v[delta_v > 0]
            neg_acc = -delta_v[delta_v < 0]

            # 特征计算
            features[i, 0] = np.mean(window)
            features[i, 1] = np.max(window)
            features[i, 2:5] = [np.mean(pos_acc), np.max(pos_acc), np.std(pos_acc)] if pos_acc.size else 0
            features[i, 5:8] = [np.mean(neg_acc), np.max(neg_acc), np.std(neg_acc)] if neg_acc.size else 0
            features[i, 8] = len(pos_acc) / window_size
            features[i, 9] = len(neg_acc) / window_size

        # 特征变换与聚类
        transformed = features @ kmeans_Tr
        clusters = np.argmin(cdist(transformed[window_size - 1:],
                                   kmeans_centers), axis=1) + 1

        return {
            "status": "success",
            "data": {
                "speed_sequence": speed_array.tolist(),
                "driving_modes": clusters.tolist(),
                "window_config": {
                    "window_size": window_size,
                    "processed_points": len(clusters)
                }
            }
        }

    except ValueError:
        raise HTTPException(400, detail="速度序列格式错误，请使用空格分隔数字")
    except Exception as e:
        raise HTTPException(500, detail=f"算法执行错误: {str(e)}")
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
