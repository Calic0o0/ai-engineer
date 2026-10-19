
import math


def calculate_total(price, quantity):
    """计算商品总价，并验证输入参数。"""

    # bool 是 int 的子类，因此要单独排除
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        raise TypeError("价格必须是数字")

    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
        raise TypeError("数量必须是数字")

    if not math.isfinite(price):
        raise ValueError("价格必须是有限数值")

    if not math.isfinite(quantity):
        raise ValueError("数量必须是有限数值")

    if price < 0:
        raise ValueError("价格不能为负数")

    if quantity < 0:
        raise ValueError("数量不能为负数")

    return price * quantity
