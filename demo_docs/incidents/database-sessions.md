# Database session exhaustion
During peak traffic the Oracle listener returned ORA-12516 because the database had no available handler for new sessions. Checkout workers accumulated connection attempts until the pool saturated.
