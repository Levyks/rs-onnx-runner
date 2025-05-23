use std::ffi::CString;
use std::fmt;
use std::os::raw::{c_char, c_int};
use std::sync::Mutex;
use tracing::{Event, Subscriber};

pub(crate) type LoggerCallback = unsafe extern "C" fn(c_int, *const c_char);

// Store the callback globally (thread-safe)
pub(crate) static LOGGER_CALLBACK: Mutex<Option<LoggerCallback>> = Mutex::new(None);

// Custom tracing subscriber
pub struct FfiSubscriber;

impl Subscriber for FfiSubscriber {
    fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
        // Enable all events for simplicity; adjust as needed
        true
    }

    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        // Minimal implementation: we don't need spans for simple logging
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {
        // No-op: we’re only handling events, not span fields
    }

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {
        // No-op
    }

    fn event(&self, event: &Event<'_>) {
        if let Ok(callback) = LOGGER_CALLBACK.lock() {
            if let Some(cb) = callback.as_ref() {
                let level = match *event.metadata().level() {
                    tracing::Level::ERROR => 1,
                    tracing::Level::WARN => 2,
                    tracing::Level::INFO => 3,
                    tracing::Level::DEBUG => 4,
                    tracing::Level::TRACE => 5,
                };

                // Format the event message
                struct MessageVisitor<'a> {
                    message: &'a mut String,
                }

                impl<'a> tracing::field::Visit for MessageVisitor<'a> {
                    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
                        if field.name() == "message" {
                            self.message.push_str(&format!("{:?}", value));
                        }
                    }
                }

                let mut message = String::new();
                let mut visitor = MessageVisitor { message: &mut message };
                event.record(&mut visitor);

                // Convert to C-compatible string
                let c_message = CString::new(message).unwrap_or_default();

                // Call the callback
                unsafe {
                    cb(level, c_message.as_ptr());
                }
            }
        }
    }

    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}