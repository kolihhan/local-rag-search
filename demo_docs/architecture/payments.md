# Payments architecture
Checkout sends authorization requests to the payment gateway through a payment worker service. Successful authorizations are persisted before order confirmation is emitted.
