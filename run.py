import uvicorn
import socket

if __name__ == "__main__":
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n🚀 FastAPI Server Starting...")
    print("Local URL:   http://127.0.0.1:8080")
    print(f"Network URL: http://{local_ip}:8080\n")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )