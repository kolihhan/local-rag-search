# Queue architecture
Asynchronous invoice and email work is written to a durable queue. Stateless workers consume jobs and scale from queue depth.
