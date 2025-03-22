# QuantumStatArb Expert Advisor

A sophisticated intraday forex trading system that implements statistical arbitrage with GPU-accelerated Kalman filtering, adaptive market microstructure analysis, and dynamic volatility regime classification.

## Features

- Statistical arbitrage with GPU-accelerated Kalman filtering
- Adaptive market microstructure analysis
- High-frequency mean reversion with quantum-inspired state detection
- Dynamic volatility regime classification
- Comprehensive cost analysis including spread and commission
- Advanced risk management with dynamic position sizing

## GPU Optimization Best Practices

### 1. Memory Management
- Efficient buffer allocation with `GPUBufferSize` parameter
- Pre-allocated arrays to minimize memory transfers
- Double buffering technique for price history updates
- Proper cleanup of GPU resources in OnDeinit()

### 2. Workload Optimization
- Configurable work group size (`GPUWorkGroupSize`) for different GPU architectures
- Parallel processing of Kalman filter states
- Minimized data transfer between CPU and GPU
- Batch processing of market data updates

### 3. Error Handling
- Graceful fallback to CPU calculations when GPU operations fail
- Comprehensive error checking for all OpenCL operations
- Safe cleanup of GPU resources
- Runtime performance monitoring

### 4. Platform Compatibility
- Conditional compilation for OpenCL support
- Support for both 32-bit and 64-bit platforms
- Dynamic selection of OpenCL device (CPU/GPU)
- Platform-specific optimizations

### 5. Performance Considerations
- Optimal work group size of 256 for most GPUs
- Minimized kernel launches by batching operations
- Efficient use of local memory in OpenCL kernels
- Vectorized operations where possible

## Installation

1. Copy the EA to your MetaTrader 5 Experts folder
2. Ensure OpenCL support is enabled on your system
3. Verify GPU drivers are up to date
4. Configure input parameters based on your trading requirements

## Input Parameters

### Trading Parameters
- RiskPercent: Risk per trade (%)
- MaxDrawdown: Maximum allowed drawdown (%)
- MaxPositions: Maximum concurrent positions
- UseGPUAcceleration: Enable/disable GPU acceleration

### GPU Settings
- GPUWorkGroupSize: Size of GPU work groups (default: 256)
- GPUBufferSize: Size of GPU buffers (default: 1024)

### Kalman Filter Parameters
- KalmanWindow: Kalman filter window size
- ProcessVariance: Process noise variance
- MeasurementVariance: Measurement noise variance

## Performance Optimization

To achieve optimal GPU performance:

1. Set appropriate GPUWorkGroupSize:
   - NVIDIA GPUs: 256 or 512
   - AMD GPUs: 256
   - Intel GPUs: 128 or 256

2. Adjust GPUBufferSize based on:
   - Available GPU memory
   - Trading frequency
   - Required historical data

3. Monitor GPU utilization:
   - Check kernel execution times
   - Monitor memory transfers
   - Verify CPU fallback triggers

## Testing

The EA includes a sophisticated OnTester() function that evaluates:
- Profit factor
- Recovery factor
- ROI
- Trade frequency
- Maximum drawdown compliance

## Requirements

- MetaTrader 5 platform
- OpenCL-capable GPU
- Updated GPU drivers
- Minimum 4GB GPU memory recommended

## License

Copyright 2024, Quantum Trading Systems. All rights reserved.

## Support

For issues and feature requests, please create an issue in the repository. 