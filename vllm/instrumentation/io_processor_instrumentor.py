# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OpenTelemetry instrumentor for IOProcessor operations."""

import functools
import time
from typing import Any

from vllm.logger import init_logger
from vllm.tracing.otel import is_otel_available

logger = init_logger(__name__)

# Version of this instrumentation library
__version__ = "0.1.0"

# Import OpenTelemetry dependencies conditionally
try:
    from opentelemetry import metrics, trace
    from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
    from opentelemetry.trace import SpanKind, StatusCode

    _IS_OTEL_AVAILABLE = True
except ImportError:
    _IS_OTEL_AVAILABLE = False
    BaseInstrumentor = object  # type: ignore
    trace = None  # type: ignore
    metrics = None  # type: ignore
    SpanKind = None  # type: ignore
    StatusCode = None  # type: ignore


def instrument():
    IOProcessorInstrumentor()._instrument()

class IOProcessorInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for IOProcessor pre_process operations.
    
    This instrumentor wraps the pre_process and pre_process_async methods
    of IOProcessor implementations to provide distributed tracing and metrics.
    
    Example usage:
        ```python
        from vllm.instrumentation.io_processor_instrumentor import IOProcessorInstrumentor
        
        # Initialize the instrumentor
        instrumentor = IOProcessorInstrumentor()
        instrumentor.instrument()
        
        # Your IOProcessor operations will now be traced
        # ...
        
        # Uninstrument when done
        instrumentor.uninstrument()
        ```
    """

    def __init__(self):
        super().__init__()
        self._original_pre_process = None
        self._original_pre_process_async = None
        self._tracer = None
        self._request_counter = None
        self._request_duration = None
        self._error_counter = None

    def instrumentation_dependencies(self):
        """Specify which version of vLLM this works with."""
        return ["vllm >= 0.6.0"]

    def _instrument(self, **kwargs):
        """Enable instrumentation by monkey-patching IOProcessor methods."""
        if not _IS_OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry is not available. IOProcessor instrumentation "
                "will not be enabled. Install with: pip install vllm[otel]"
            )
            return

        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")

        # Create a tracer for this instrumentation
        self._tracer = trace.get_tracer(
            instrumenting_module_name="vllm.plugins.io_processors",
            instrumenting_library_version=__version__,
            tracer_provider=tracer_provider,
        )

        # Create metrics
        meter = metrics.get_meter(
            name="vllm.io_processor",
            version=__version__,
            meter_provider=meter_provider,
        )

        self._request_counter = meter.create_counter(
            name="io_processor.pre_process.count",
            description="Total number of IOProcessor pre_process operations",
            unit="operations",
        )

        self._request_duration = meter.create_histogram(
            name="io_processor.pre_process.duration",
            description="Duration of IOProcessor pre_process operations",
            unit="ms",
        )

        self._error_counter = meter.create_counter(
            name="io_processor.pre_process.errors",
            description="Total number of IOProcessor pre_process errors",
            unit="errors",
        )

        # Monkey-patch the IOProcessor methods
        from vllm.plugins.io_processors.interface import IOProcessor

        self._original_pre_process = IOProcessor.pre_process
        self._original_pre_process_async = IOProcessor.pre_process_async

        IOProcessor.pre_process = self._instrumented_pre_process
        IOProcessor.pre_process_async = self._instrumented_pre_process_async

        logger.info("IOProcessor instrumentation enabled")

    def _uninstrument(self, **kwargs):
        """Disable instrumentation by restoring original methods."""
        if not _IS_OTEL_AVAILABLE or self._original_pre_process is None:
            return

        from vllm.plugins.io_processors.interface import IOProcessor

        IOProcessor.pre_process = self._original_pre_process
        IOProcessor.pre_process_async = self._original_pre_process_async

        self._original_pre_process = None
        self._original_pre_process_async = None

        logger.info("IOProcessor instrumentation disabled")

    def _instrumented_pre_process(self, processor_self, prompt, request_id=None, **kwargs):
        """Wrapped version of pre_process that creates spans and metrics."""
        # Get processor class name for better span naming
        processor_class = processor_self.__class__.__name__
        span_name = f"IOProcessor.pre_process [{processor_class}]"

        # Build span attributes
        attributes = {
            "io_processor.class": processor_class,
            "io_processor.method": "pre_process",
        }
        if request_id:
            attributes["io_processor.request_id"] = request_id

        # Add prompt type information if available
        if hasattr(prompt, "__class__"):
            attributes["io_processor.prompt_type"] = prompt.__class__.__name__

        # Start a span for this operation
        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
            attributes=attributes,
        ) as span:
            start_time = time.time()
            error_occurred = False

            try:
                # Call the original pre_process method
                result = self._original_pre_process(
                    processor_self, prompt, request_id, **kwargs
                )

                # Record success status
                span.set_status(StatusCode.OK)

                # Add result information to span
                if isinstance(result, (list, tuple)):
                    span.set_attribute("io_processor.result_count", len(result))

                return result

            except Exception as exc:
                # Record the exception on the span
                error_occurred = True
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

            finally:
                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                metric_attrs = {
                    "processor_class": processor_class,
                    "method": "pre_process",
                }

                self._request_counter.add(1, metric_attrs)
                self._request_duration.record(duration_ms, metric_attrs)

                if error_occurred:
                    self._error_counter.add(1, metric_attrs)

                # Add duration to span
                span.set_attribute("io_processor.duration_ms", duration_ms)

    def _instrumented_pre_process_async(
        self, processor_self, prompt, request_id=None, **kwargs
    ):
        """Wrapped version of pre_process_async that creates spans and metrics."""
        # Get processor class name for better span naming
        processor_class = processor_self.__class__.__name__
        span_name = f"IOProcessor.pre_process_async [{processor_class}]"

        # Build span attributes
        attributes = {
            "io_processor.class": processor_class,
            "io_processor.method": "pre_process_async",
        }
        if request_id:
            attributes["io_processor.request_id"] = request_id

        # Add prompt type information if available
        if hasattr(prompt, "__class__"):
            attributes["io_processor.prompt_type"] = prompt.__class__.__name__

        # Create an async wrapper
        @functools.wraps(self._original_pre_process_async)
        async def async_wrapper():
            # Start a span for this operation
            with self._tracer.start_as_current_span(
                name=span_name,
                kind=SpanKind.INTERNAL,
                attributes=attributes,
            ) as span:
                start_time = time.time()
                error_occurred = False

                try:
                    # Call the original pre_process_async method
                    result = await self._original_pre_process_async(
                        processor_self, prompt, request_id, **kwargs
                    )

                    # Record success status
                    span.set_status(StatusCode.OK)

                    # Add result information to span
                    if isinstance(result, (list, tuple)):
                        span.set_attribute("io_processor.result_count", len(result))

                    return result

                except Exception as exc:
                    # Record the exception on the span
                    error_occurred = True
                    span.set_status(StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise

                finally:
                    # Record metrics
                    duration_ms = (time.time() - start_time) * 1000
                    metric_attrs = {
                        "processor_class": processor_class,
                        "method": "pre_process_async",
                    }

                    self._request_counter.add(1, metric_attrs)
                    self._request_duration.record(duration_ms, metric_attrs)

                    if error_occurred:
                        self._error_counter.add(1, metric_attrs)

                    # Add duration to span
                    span.set_attribute("io_processor.duration_ms", duration_ms)

        return async_wrapper()


# Global instrumentor instance
_instrumentor = None


def instrument_io_processor(**kwargs):
    """Convenience function to instrument IOProcessor operations.
    
    Args:
        **kwargs: Optional tracer_provider and meter_provider arguments.
    
    Returns:
        The IOProcessorInstrumentor instance.
    """
    global _instrumentor
    
    if not is_otel_available():
        logger.warning(
            "OpenTelemetry is not available. IOProcessor instrumentation "
            "will not be enabled. Install with: pip install vllm[otel]"
        )
        return None
    
    if _instrumentor is None:
        _instrumentor = IOProcessorInstrumentor()
    
    _instrumentor._instrument(**kwargs)
    return _instrumentor


def uninstrument_io_processor():
    """Convenience function to uninstrument IOProcessor operations."""
    global _instrumentor
    
    if _instrumentor is not None:
        _instrumentor.uninstrument()
