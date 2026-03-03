# vLLM Instrumentation

This package provides OpenTelemetry instrumentation for vLLM components.

## IOProcessor Instrumentor

The `IOProcessorInstrumentor` provides distributed tracing and metrics for IOProcessor operations, specifically wrapping the `pre_process` and `pre_process_async` methods.

### Features

- **Distributed Tracing**: Creates spans for each IOProcessor pre_process operation
- **Metrics Collection**: Records operation counts, durations, and error rates
- **Automatic Context Propagation**: Integrates with vLLM's existing OpenTelemetry setup
- **Async Support**: Works with both sync and async pre_process methods

### Installation

Install vLLM with OpenTelemetry support:

```bash
pip install vllm[otel]
```

### Usage

#### Basic Usage

```python
from vllm.instrumentation import instrument_io_processor, uninstrument_io_processor

# Enable instrumentation
instrument_io_processor()

# Your IOProcessor operations will now be traced
# ... use vLLM with IOProcessor plugins ...

# Disable instrumentation when done
uninstrument_io_processor()
```

#### With Custom Providers

```python
from opentelemetry import trace, metrics
from vllm.instrumentation import IOProcessorInstrumentor

# Create custom providers
tracer_provider = trace.get_tracer_provider()
meter_provider = metrics.get_meter_provider()

# Initialize and instrument
instrumentor = IOProcessorInstrumentor()
instrumentor.instrument(
    tracer_provider=tracer_provider,
    meter_provider=meter_provider
)

# ... use vLLM ...

# Uninstrument
instrumentor.uninstrument()
```

#### Integration with vLLM Server

When running vLLM with OpenTelemetry tracing enabled, the IOProcessor instrumentor can be automatically enabled:

```python
from vllm import LLM
from vllm.instrumentation import instrument_io_processor

# Enable IOProcessor instrumentation
instrument_io_processor()

# Initialize vLLM with tracing
llm = LLM(
    model="your-model",
    # ... other config ...
)

# IOProcessor operations will now be traced
```

### Collected Metrics

The instrumentor collects the following metrics:

1. **io_processor.pre_process.count** (Counter)
   - Total number of pre_process operations
   - Labels: `processor_class`, `method`

2. **io_processor.pre_process.duration** (Histogram)
   - Duration of pre_process operations in milliseconds
   - Labels: `processor_class`, `method`

3. **io_processor.pre_process.errors** (Counter)
   - Total number of pre_process errors
   - Labels: `processor_class`, `method`

### Span Attributes

Each span includes the following attributes:

- `io_processor.class`: The IOProcessor implementation class name
- `io_processor.method`: The method being traced (`pre_process` or `pre_process_async`)
- `io_processor.request_id`: The request ID (if provided)
- `io_processor.prompt_type`: The type of prompt being processed
- `io_processor.result_count`: Number of results returned (for sequence results)
- `io_processor.duration_ms`: Operation duration in milliseconds

### Example with OpenTelemetry Collector

```python
import os
from vllm.instrumentation import instrument_io_processor

# Configure OpenTelemetry endpoint
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:4317"
os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "grpc"

# Enable instrumentation
instrument_io_processor()

# Your vLLM application code here
# Traces will be exported to the configured endpoint
```

### Viewing Traces

Traces can be viewed in any OpenTelemetry-compatible backend such as:

- Jaeger
- Zipkin
- Grafana Tempo
- Honeycomb
- Datadog
- New Relic

Example span hierarchy:
```
IOProcessor.pre_process [PrithviMultimodalDataProcessor]
├── Duration: 45.2ms
├── Attributes:
│   ├── io_processor.class: PrithviMultimodalDataProcessor
│   ├── io_processor.method: pre_process
│   ├── io_processor.request_id: req_123
│   ├── io_processor.prompt_type: ImagePrompt
│   ├── io_processor.result_count: 4
│   └── io_processor.duration_ms: 45.2
└── Status: OK
```

### Troubleshooting

**Instrumentation not working:**
- Ensure OpenTelemetry packages are installed: `pip install vllm[otel]`
- Check that instrumentation is called before IOProcessor operations
- Verify OpenTelemetry is properly configured in your environment

**No traces appearing:**
- Verify `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set correctly
- Check that your OpenTelemetry collector/backend is running
- Enable debug logging to see trace export attempts

**Type errors in IDE:**
- These are expected when OpenTelemetry is not installed
- The instrumentor gracefully handles missing dependencies at runtime

### API Reference

#### `instrument_io_processor(**kwargs)`

Convenience function to instrument IOProcessor operations.

**Parameters:**
- `tracer_provider` (optional): Custom tracer provider
- `meter_provider` (optional): Custom meter provider

**Returns:**
- `IOProcessorInstrumentor` instance or `None` if OpenTelemetry is unavailable

#### `uninstrument_io_processor()`

Convenience function to remove IOProcessor instrumentation.

#### `IOProcessorInstrumentor`

The main instrumentor class implementing the OpenTelemetry BaseInstrumentor interface.

**Methods:**
- `instrument(**kwargs)`: Enable instrumentation
- `uninstrument(**kwargs)`: Disable instrumentation
- `instrumentation_dependencies()`: Returns list of dependencies

### Contributing

When adding new instrumentation:

1. Follow the existing pattern in `io_processor_instrumentor.py`
2. Add comprehensive span attributes
3. Include both sync and async support where applicable
4. Add metrics for operation counts, durations, and errors
5. Update this README with usage examples

### License

SPDX-License-Identifier: Apache-2.0