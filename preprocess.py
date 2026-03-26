def preprocess(data):
    return{
        "heart_rate":
        float(data["heart_rate"]),
        "temp": float(data["temp"]),
        "oxygen": float(data["oxygen"])
    }