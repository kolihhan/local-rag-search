# API latency after deployment
A deployment changed a cache key and sharply reduced hit rate. The API became slow under production load because requests fell through to the database. Rolling back the cache change restored latency.
