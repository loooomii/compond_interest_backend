import numpy as np
from numpy import zeros, ndarray
from numpy import round as np_round
from dataclasses import dataclass
from pandas import date_range, DataFrame, Timestamp
import math
from typing import Literal, Dict

# taku only
from matplotlib import pyplot as plt


@dataclass
class InvestmentResult:
    """investment result dataclass"""
    final_balance: list[float | int]
    total_principal: float | int
    total_return: list[float | int]
    monthly_data: DataFrame


def _validate_inputs(params: Dict) -> None:
    """Validate input parameters."""
    if params['y_return'] < 0:
        raise ValueError("Yearly return rate cannot be less than 0")
    if params['horizon'] <= 0:
        raise ValueError("Investment horizon must be positive")
    if params['m_investment'] < 0:
        raise ValueError("Monthly investment cannot be negative")
    if params['init_balance'] < 0:
        raise ValueError("Initial balance cannot be negative")
    if params['method'] not in ["geometric", "arithmetic"]:
        raise ValueError("Method must be either 'geometric' or 'arithmetic'")
    if not isinstance(params['increment'], (int, float)):
        raise ValueError("Increment amount must be a number")
    if params['incre_period'] < 0:
        raise ValueError("Increment period cannot be negative")


class InvestmentCalculator:
    """
    A class to calculate the investment return and investment plan.

     Parameters:
        y_return (float): Yearly return rate (as decimal, e.g., 0.1 for 10%)
        y_risk (float): Yearly risk rate (as decimal, e.g., 0.1 for 10%)
        horizon (int): Investment horizon in years
        m_investment (float): Monthly investment amount
        init_balance (float): Initial balance (default: 0)
        method (str): Method to calculate monthly return - "geometric" or "arithmetic" (default: "geometric")
        increment (float): Annual increment to monthly investment. Defaults to 0.
        incre_period (int): Number of years to apply increment. Defaults to 0.
    """

    def __init__(self, y_return: float = 0,
                 horizon: int = 1,
                 m_investment: float = 0,
                 init_balance: float = 0,
                 method: Literal["geometric", "arithmetic"] = "geometric",
                 increment: float = 0,
                 incre_period: int = 0,
                 y_risk: float = 0):

        # Input validation
        _validate_inputs(locals())

        # Protected attributes in this class (suggest to use in subclasses and this class)
        self._y_return = y_return
        self._horizon = horizon
        self._m_investment = m_investment
        self._init_balance = init_balance
        self._method = method
        self._increment = increment
        self._increment_period = incre_period

        # Protected attributes to be used in this class only
        self._monthly_return = self._calculate_monthly_return()
        self._monthly_risk = y_risk / np.sqrt(12)

    @property
    def y_return(self) -> float:
        """yearly return rate"""
        return self._y_return

    @property
    def horizon(self) -> int:
        """investment horizon in years"""
        return self._horizon

    @property
    def m_investment(self) -> float:
        """monthly investment amount"""
        return self._m_investment

    @property
    def init_balance(self) -> float:
        """initial balance"""
        return self._init_balance

    @staticmethod
    def generate_confidence_interval(month_num: int,
                                     monthly_rate: float,
                                     monthly_risk: float,
                                     simulation_times: int = 100,
                                     df: int = 8,
                                     distribution: str = "Student - t") -> ndarray:
        """
        Generate a confidence interval for the investment return.
        最后只保留2.5%分位数，均值和97.5%分位数，所以只保留3列
        df: 自由度
        simulation_times: 模拟次数
        """
        # 创建一个三维数组来存储所有模拟结果
        # 形状为 (simulation_times, month_num, 1)
        if monthly_risk == 0:
            # 如果没有风险，直接返回确定值
            return np.zeros(shape=(month_num, 3)) + monthly_rate

        if distribution not in ["Student - t", "Normal"]:
            raise ValueError("Distribution must be either 'Student - t' or 'Normal'")
        # 使用t分布进行模拟
        if distribution == "Student - t":
            all_simulations = monthly_rate + monthly_risk * np.random.standard_t(
                df=df,
                size=(simulation_times, month_num, 1)
            )
        # 使用正态分布进行模拟
        elif distribution == "Normal":
            all_simulations = monthly_rate + monthly_risk * np.random.normal(
                size=(simulation_times, month_num, 1)
            )

        # 计算每个月的统计量
        percentile_2_5 = np.percentile(all_simulations, 2.5, axis=0)
        mean_values = np.mean(all_simulations, axis=0)
        percentile_97_5 = np.percentile(all_simulations, 97.5, axis=0)

        # 组合结果
        result = np.concatenate([
            percentile_2_5,
            mean_values,
            percentile_97_5
        ], axis=1)

        return result

    def _calculate_monthly_return(self,
                                  annual_return: float = None) -> float:
        """Calculate the monthly return of an expected yearly return."""
        if self._method == "geometric":
            return pow(1 + self.y_return, 1 / 12) - 1 if annual_return is None else pow(1 + annual_return, 1 / 12) - 1
        elif self._method == "arithmetic":
            return self.y_return / 12 if annual_return is None else annual_return / 12

    def automatic_investment(self,
                             horizon: float = None,
                             m_investment: float = None,
                             annual_rate: float = None,
                             annual_risk: float = None) -> InvestmentResult:
        """
        Calculate the final balance with optional periodic investment increment.

        假设在每个月的月初进行定投，每月定投额为m_investment，年收益率为y_return，投资期为horizon年。
        因此，每个月月初的余额计算公式为：balance = balance * (1 + monthly_return) + m_investment
        这里，balance为上个月月初的余额，这一部分会享受一整个月的收益。

        是否增加定投额的判断
        1.是否经过了一年：(i+1) % 12 == 0
        2.增加定投额是否为0：increment != 0
        3.当前年限是否在增加定投额年限内：year_num <= incre_period
        4.判断是否为第一年，第一年不进行增加定投额的操作，增加定投额从第二年开始：year_num != 0

        Parameters:三个可选参数主要用于使用back_to_present方法计算完所需的值后，再次调用automatic_investment方法
            1.horizon (float): Investment horizon in years (default: None, use class attribute)
            2.m_investment (float): Monthly investment amount (default: None, use class attribute)
            3.monthly_return (float): Monthly return rate (default: None, use class attribute)
            :return: InvestmentResult
        Returns:
            float: Final balance after investment horizon.
        """

        """计算投资期数以及设置当前月投资额和期望收益率"""
        monthly_return = self._monthly_return \
            if annual_rate is None else self._calculate_monthly_return(annual_return=annual_rate)  # 判断是否有传入年化收益率
        monthly_risk = self._monthly_risk if annual_risk is None else annual_risk / np.sqrt(12)  # 判断是否有传入年化风险率
        month_num = self.horizon * 12 if horizon is None else horizon * 12
        current_monthly_investment = self.m_investment if m_investment is None else m_investment
        excepted_return = self.generate_confidence_interval(month_num=month_num,
                                                            monthly_rate=monthly_return,
                                                            monthly_risk=monthly_risk)

        """initial data array"""
        """第一个元素为初始值，后续元素开始依次为投资一个月，两个月，三个月……时的月初时候的数值"""
        balances = zeros(shape=(month_num + 1, 3))  # 账户余额
        principals = zeros(month_num + 1)  # 投入本金
        returns = zeros(shape=(month_num + 1, 3))  # 投资收益
        investment_amount = zeros(month_num + 1)  # 投资额

        """setting initial value"""
        """初始余额设置为0的情况下，会默认为每月定投额"""
        balances[0] = self.init_balance
        principals[0] = self.init_balance
        investment_amount[0] = self.init_balance

        for i in range(month_num):
            year_num = i // 12

            # Apply increment to monthly investment or not
            if i % 12 == 0 and self._increment != 0 and \
                    year_num <= self._increment_period and year_num != 0:
                current_monthly_investment += self._increment

            balances[i + 1] = (balances[i] * (1 + excepted_return[i]) +
                               current_monthly_investment)
            principals[i + 1] = principals[i] + current_monthly_investment
            returns[i + 1] = balances[i + 1] - principals[i + 1]
            investment_amount[i + 1] = current_monthly_investment

        """Create monthly data"""
        dates = date_range(
            start=Timestamp.now().to_period("M").to_timestamp(),
            periods=month_num + 1,
            freq='ME'
        )

        monthly_data = DataFrame({
            "Date": dates,
            "Principal": np_round(principals).astype(int),
            "2.5% Return": np_round(returns[:, 0]).astype(int),  # 2.5%低收益
            "Return": np_round(returns[:, 1]).astype(int),
            "97.5% Return": np_round(returns[:, 2]).astype(int),
            "2.5% Balance": np_round(balances[:, 0]).astype(int),  # 2.5%低余额
            "Balance": np_round(balances[:, 1]).astype(int),
            "97.5% Balance": np_round(balances[:, 2]).astype(int),
            "Investment": np_round(investment_amount).astype(int)
        })

        return InvestmentResult(
            final_balance=[  # 返回一个包含三个值的列表
                round(balances[-1, 0].item()),  # 2.5%分位数的最终余额
                round(balances[-1, 1].item()),  # 均值情况的最终余额
                round(balances[-1, 2].item())  # 97.5%分位数的最终余额
            ],
            total_principal=round(principals[-1].item()),  # principals是一维的，所以这里没问题
            total_return=[  # 返回一个包含三个值的列表
                round(returns[-1, 0].item()),  # 2.5%分位数的总收益
                round(returns[-1, 1].item()),  # 均值情况的总收益
                round(returns[-1, 2].item())  # 97.5%分位数的总收益
            ],
            monthly_data=monthly_data
        )

    def back_to_present(self,
                        target: Literal["amount", "rate", "horizon"],
                        target_value: float,
                        initial: float = None) -> float | ndarray:
        """
        Calculate either required monthly investment or required monthly return
        to reach a target value.
        除了计算出的收益率之外，其余回传的所有结果会向上取整
        除了要传入的数值之外，其余的数值都会使用类中的属性值，如果需要传入其他数值，需要更改类的属性值。

        Parameters:
            target (str): "num" for monthly investment or "rate" for required return
            or "horizon" for required investment horizon.
            value_target (float): Target final balance.

        Returns:
            float: Required monthly investment or monthly return rate
            :param target_value: 目标金额
            :param target: 所求目标类型，amount为每月投资额，rate为年化收益率，horizon为投资期限
            :param initial: 初始值
        """
        #  TODO: 增加其他数值的传入，让用户可以传入其他数值而不用修改类的属性值。
        if target_value <= 0:
            raise ValueError("Target value must be positive")

        month_num = self.horizon * 12
        initial_balance = self.init_balance if initial is None else initial

        if target == "amount":
            # Calculate required monthly investment
            if target_value <= initial_balance:
                return 0  # 已达到目标,无需投资
            # 等比数列求和
            if self._monthly_return == 0:
                # Special case for 0% return
                amount = (target_value - initial_balance) / month_num
                return math.ceil(amount)
            else:
                numerator = (target_value - initial_balance * pow(1 + self._monthly_return, month_num))
                denominator = (pow(1 + self._monthly_return, month_num) - 1) / self._monthly_return
                amount = numerator / denominator
                return math.ceil(amount)

        elif target == "rate":
            # Calculate required monthly return rate using numerical method
            from scipy.optimize import fsolve
            if initial_balance == 0 and self.m_investment == 0:  # 检测投资额和初始资产不能同时为0
                return 0
            if target_value <= initial_balance + self.m_investment * month_num:  # 如果目标值小于初始值+总投资额，直接返回0
                return 0

            tolerance = 1e-6  # 设定精度，用于判断是否达到目标值
            left, right = 0, 10.0  # 设定二分法的左右边界，因为要求预期收益率大于0，所以左边界为0（小于0时候没有投资的必要）

            def calc_final_value(r):
                if abs(r) < 1e-10:
                    return initial_balance + self.m_investment * month_num
                return (initial_balance * pow(1 + r, month_num) +
                        self.m_investment * (pow(1 + r, month_num) - 1) / r)

            while right - left > tolerance:
                mid = (left + right) / 2
                final_value = calc_final_value(mid)

                if (final_value - target_value < tolerance) and final_value >= target_value:
                    return mid
                elif final_value < target_value:
                    left = mid
                else:
                    right = mid

            monthly = (left + right) / 2
            annual = pow(1 + monthly, 12) - 1

            return max(annual, 0)

        elif target == "horizon":
            # Calculate required investment horizon using logarithm formula
            if target_value <= initial_balance:  # 如果目标值小于初始值，直接返回0
                return 0  # 已达到目标
            # Calculate number of months using the derived formula
            if self._monthly_return == 0:
                # Special case for 0% return
                months = (target_value - initial_balance) / self.m_investment
            else:
                numerator = target_value + self.m_investment / self._monthly_return
                denominator = initial_balance + self.m_investment / self._monthly_return
                months = math.log(numerator / denominator) / math.log(1 + self._monthly_return)

            # Convert months to years, ensure non-negative and ceiling to next integer
            years = math.ceil(max(0, months / 12))
            return years


if __name__ == "__main__":
    # 使用示例
    try:
        # 创建计算器实例：10%年收益率，5年投资期，每月投资1000元
        calc = InvestmentCalculator(
            y_return=0.1,  # 10% 年收益率
            horizon=10,  # 5年投资期
            m_investment=10,  # 每月投资1000元
            init_balance=0,  # 初始余额0元
            method="geometric",  # 使用几何平均值计算月收益率
            increment=0,  # 每年增加100元投资
            incre_period=0,  # 持续3年增
            y_risk=0.1
        )

        # 计算最终余额
        final_result = calc.automatic_investment()

        # 修改后的代码（正确显示三种情况）
        print(f"Final balance after {calc.horizon} years:")
        print(f"  - Conservative (2.5%): {final_result.final_balance[0]:.2f}")
        print(f"  - Expected (Mean): {final_result.final_balance[1]:.2f}")
        print(f"  - Optimistic (97.5%): {final_result.final_balance[2]:.2f}")

        # 同样地，我们也需要修改其他打印语句
        print("\nTotal return:")
        print(f"  - Conservative (2.5%): {final_result.total_return[0]:.2f}")
        print(f"  - Expected (Mean): {final_result.total_return[1]:.2f}")
        print(f"  - Optimistic (97.5%): {final_result.total_return[2]:.2f}")

        print(f"\nTotal principal invested: {final_result.total_principal:.2f}")

        result = calc.automatic_investment()
        print(f"保守情况(2.5%)的最终余额: {result.final_balance[0]}")
        print(f"预期情况的最终余额: {result.final_balance[1]}")
        print(f"乐观情况(97.5%)的最终余额: {result.final_balance[2]}")

        # 画图
        final_result.monthly_data.plot(x="Date", y=["Principal", "2.5% Return", "Return", "97.5% Return", "Balance",
                                                    "2.5% Balance", "97.5% Balance"], )
        # taku only
        plt.show()

        # 计算达到目标所需的每月投资额
        target_value = 1000
        required_monthly = calc.back_to_present("amount", target_value)
        print(f"Required monthly investment to reach {target_value} "
              f"with {calc.y_return} monthly return "
              f"and {calc.init_balance} initial balance : {required_monthly:.2f}")

        # 计算达到目标所需的年化收益率
        required_return = calc.back_to_present("rate", target_value)
        print(f"Reaching the {target_value} target with {calc.m_investment} monthly investment, "
              f"required yearly return rate: {required_return:.4%}")

        # 计算达到目标所需的投资期限
        required_horizon = calc.back_to_present("horizon", target_value)
        print(f"Reaching the {target_value} target with {calc.m_investment} monthly investment, "
              f"required investment horizon: {required_horizon} years")

    except ValueError as e:
        print(f"Error: {e}")
