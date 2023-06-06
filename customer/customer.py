class Customer:
    def __init__(self, customer_id, data):
        self.customer_id = customer_id
        self.data = data

    def __str__(self):
        return f"Customer {self.customer_id}: {self.data}"
    
