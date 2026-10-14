FROM python:3.12-slim
WORKDIR /app
COPY day26/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY day23/ ./day23/
COPY day25/ ./day25/
COPY day26/ ./day26/
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["python", "day26/app.py", "--host", "0.0.0.0", "--port", "8000"]
