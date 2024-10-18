FROM python:3.11-slim AS build

# Copy the application code and configuration
ADD . /

# Set environment variables
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH=.

EXPOSE 5021 5022 5023 5024 5025 5026 5027 5028 5029 5030 5031 5032 5033 5034

# Install Python packages
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the remaining files
COPY . .

# Set the container command
CMD [ "python3", "./pool_modbus_emulator.py" ]
