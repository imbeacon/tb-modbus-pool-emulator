# ThingsBoard Modbus Pool Emulator

This is a simple modbus pool emulator that can be used to test Modbus integration with platform using ThingsBoard IoT Gateway.

## Installation

1. Pull emulator docker image:
    ```bash
    docker pull thingsboard/tb-modbus-pool-emulator:latest
    ```
2. Run the emulator using the following command, which will start the emulator on ports 5021-5034:
    ```bash
    docker run --rm -d --name tb-modbus-pool-emulator -p 5021-5034:5021-5034 tb-modbus-pool-emulator
    ```
   ***Note***: *If you run the gateway first - it may take up to 2 minutes since the emulator starts to the gateway connects to it*.
3. Create a new gateway device in ThingsBoard and copy the access token.
4. Pull gateway image from Dockerhub using the following command:
    ```bash
    docker pull thingsboard/tb-gateway:latest
    ```
4. Replace YOUR_ACCESS_TOKEN with the access token of the gateway device and host (if you want to connect to ThingsBoard, not on your machine) in docker-compose.yml.
5. Run the gateway using the following command:
    ```bash
    docker-compose up
    ```
6. Add Modbus connector and configure the ThingsBoard IoT Gateway on UI to connect to the emulated devices. You can find configuration in *pool_connector.json*.
7. The gateway connects to emulated devices, creates them on the platform, starts receive data from devices and send it to ThingsBoard.
