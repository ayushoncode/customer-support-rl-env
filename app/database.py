from typing import Dict, Any

ORDER_DB: Dict[str, Dict[str, Any]] = {
    "ORD-4521": {"customer": "Ayush Raj", "item": "Wireless Headphones", "amount": 89.99, "status": "delayed", "days_since_order": 18, "eligible_refund": True},
    "ORD-7823": {"customer": "Priya Sharma", "item": "Laptop Stand", "amount": 34.99, "status": "delivered", "days_since_order": 45, "eligible_refund": False},
    "ORD-9921": {"customer": "Rahul Mehta", "item": "Dell Laptop", "amount": 1200.00, "status": "lost", "days_since_order": 21, "eligible_refund": True},
    "ORD-3310": {"customer": "Sara Khan", "item": "USB Hub", "amount": 24.99, "status": "wrong_item", "days_since_order": 3, "eligible_refund": True},
    "ORD-6612": {"customer": "John Doe", "item": "Monitor", "amount": 399.99, "status": "delivered", "days_since_order": 5, "eligible_refund": True},
}

POLICY = {
    "refund_window_days": 30,
    "max_refund_amount": 500.00,
    "escalation_threshold": 200.00,
    "fraud_auto_block": True,
    "replacement_eligible_statuses": ["lost", "wrong_item", "damaged"],
}

def lookup_order(order_id: str) -> Dict[str, Any]:
    return ORDER_DB.get(order_id, {"error": "Order not found"})

def check_refund_eligible(order_id: str) -> Dict[str, Any]:
    order = ORDER_DB.get(order_id)
    if not order:
        return {"eligible": False, "reason": "Order not found"}
    if order["days_since_order"] > POLICY["refund_window_days"]:
        return {"eligible": False, "reason": f"Outside {POLICY['refund_window_days']}-day refund window"}
    if order["amount"] > POLICY["max_refund_amount"]:
        return {"eligible": False, "reason": "Amount exceeds max refund limit", "escalate": True}
    return {"eligible": order["eligible_refund"], "reason": "Within policy"}
