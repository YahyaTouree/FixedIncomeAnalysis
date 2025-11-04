"""
Binomial Tree Option Pricing Model

Implementation of the Cox-Ross-Rubinstein binomial tree model for
American and European options with early exercise capability.
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings

class BinomialTree:
    """
    Cox-Ross-Rubinstein binomial tree option pricing model.

    This class implements the binomial tree method for pricing both
    European and American options, with support for dividends and
    early exercise optimization.
    """

    def __init__(self):
        self convergence_threshold = 1e-6

    def calculate_tree_parameters(self, S: float, K: float, T: float,
                                 r: float, sigma: float, n_steps: int,
                                 q: float = 0.0) -> Tuple[float, float, float, float]:
        """
        Calculate binomial tree parameters.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            n_steps: Number of time steps
            q: Dividend yield

        Returns:
            Tuple of (dt, u, d, p) parameters
        """
        dt = T / n_steps

        # Up and down factors
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u

        # Risk-neutral probability
        p = (np.exp((r - q) * dt) - d) / (u - d)

        # Ensure probability is in valid range
        p = max(0.0, min(1.0, p))

        return dt, u, d, p

    def european_call(self, S: float, K: float, T: float,
                     r: float, sigma: float, n_steps: int,
                     q: float = 0.0) -> float:
        """
        Price European call option using binomial tree.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            n_steps: Number of time steps
            q: Dividend yield

        Returns:
            European call option price
        """
        dt, u, d, p = self.calculate_tree_parameters(S, K, T, r, sigma, n_steps, q)
        discount = np.exp(-r * dt)

        # Initialize option values at expiration
        option_values = np.zeros(n_steps + 1)

        for i in range(n_steps + 1):
            stock_price = S * (u ** i) * (d ** (n_steps - i))
            option_values[i] = max(stock_price - K, 0)

        # Backward induction
        for step in range(n_steps - 1, -1, -1):
            for i in range(step + 1):
                option_values[i] = discount * (
                    p * option_values[i + 1] + (1 - p) * option_values[i]
                )

        return option_values[0]

    def european_put(self, S: float, K: float, T: float,
                    r: float, sigma: float, n_steps: int,
                    q: float = 0.0) -> float:
        """
        Price European put option using binomial tree.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            n_steps: Number of time steps
            q: Dividend yield

        Returns:
            European put option price
        """
        dt, u, d, p = self.calculate_tree_parameters(S, K, T, r, sigma, n_steps, q)
        discount = np.exp(-r * dt)

        # Initialize option values at expiration
        option_values = np.zeros(n_steps + 1)

        for i in range(n_steps + 1):
            stock_price = S * (u ** i) * (d ** (n_steps - i))
            option_values[i] = max(K - stock_price, 0)

        # Backward induction
        for step in range(n_steps - 1, -1, -1):
            for i in range(step + 1):
                option_values[i] = discount * (
                    p * option_values[i + 1] + (1 - p) * option_values[i]
                )

        return option_values[0]

    def american_call(self, S: float, K: float, T: float,
                     r: float, sigma: float, n_steps: int,
                     q: float = 0.0) -> float:
        """
        Price American call option using binomial tree with early exercise.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            n_steps: Number of time steps
            q: Dividend yield

        Returns:
            American call option price
        """
        dt, u, d, p = self.calculate_tree_parameters(S, K, T, r, sigma, n_steps, q)
        discount = np.exp(-r * dt)

        # Initialize stock prices and option values
        stock_prices = np.zeros((n_steps + 1, n_steps + 1))
        option_values = np.zeros((n_steps + 1, n_steps + 1))

        # Calculate stock prices at each node
        for i in range(n_steps + 1):
            for j in range(i + 1):
                stock_prices[j, i] = S * (u ** j) * (d ** (i - j))

        # Initialize option values at expiration
        for i in range(n_steps + 1):
            option_values[i, n_steps] = max(stock_prices[i, n_steps] - K, 0)

        # Backward induction with early exercise check
        for step in range(n_steps - 1, -1, -1):
            for i in range(step + 1):
                # Option value if held
                hold_value = discount * (
                    p * option_values[i + 1, step + 1] +
                    (1 - p) * option_values[i, step + 1]
                )

                # Option value if exercised early
                exercise_value = max(stock_prices[i, step] - K, 0)

                # American option value
                option_values[i, step] = max(hold_value, exercise_value)

        return option_values[0, 0]

    def american_put(self, S: float, K: float, T: float,
                    r: float, sigma: float, n_steps: int,
                    q: float = 0.0) -> float:
        """
        Price American put option using binomial tree with early exercise.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            n_steps: Number of time steps
            q: Dividend yield

        Returns:
            American put option price
        """
        dt, u, d, p = self.calculate_tree_parameters(S, K, T, r, sigma, n_steps, q)
        discount = np.exp(-r * dt)

        # Initialize stock prices and option values
        stock_prices = np.zeros((n_steps + 1, n_steps + 1))
        option_values = np.zeros((n_steps + 1, n_steps + 1))

        # Calculate stock prices at each node
        for i in range(n_steps + 1):
            for j in range(i + 1):
                stock_prices[j, i] = S * (u ** j) * (d ** (i - j))

        # Initialize option values at expiration
        for i in range(n_steps + 1):
            option_values[i, n_steps] = max(K - stock_prices[i, n_steps], 0)

        # Backward induction with early exercise check
        for step in range(n_steps - 1, -1, -1):
            for i in range(step + 1):
                # Option value if held
                hold_value = discount * (
                    p * option_values[i + 1, step + 1] +
                    (1 - p) * option_values[i, step + 1]
                )

                # Option value if exercised early
                exercise_value = max(K - stock_prices[i, step], 0)

                # American option value
                option_values[i, step] = max(hold_value, exercise_value)

        return option_values[0, 0]

    def calculate_greeks(self, S: float, K: float, T: float,
                        r: float, sigma: float, n_steps: int = 100,
                        option_type: str = 'call', style: str = 'american',
                        q: float = 0.0) -> dict:
        """
        Calculate Greeks using finite difference method.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free interest rate
            sigma: Volatility
            n_steps: Number of tree steps
            option_type: 'call' or 'put'
            style: 'american' or 'european'
            q: Dividend yield

        Returns:
            Dictionary containing Greeks
        """
        # Base price
        if style.lower() == 'american':
            if option_type.lower() == 'call':
                base_price = self.american_call(S, K, T, r, sigma, n_steps, q)
            else:
                base_price = self.american_put(S, K, T, r, sigma, n_steps, q)
        else:
            if option_type.lower() == 'call':
                base_price = self.european_call(S, K, T, r, sigma, n_steps, q)
            else:
                base_price = self.european_put(S, K, T, r, sigma, n_steps, q)

        # Delta (first-order price sensitivity)
        dS = S * 0.01
        if style.lower() == 'american':
            if option_type.lower() == 'call':
                price_up = self.american_call(S + dS, K, T, r, sigma, n_steps, q)
                price_down = self.american_call(S - dS, K, T, r, sigma, n_steps, q)
            else:
                price_up = self.american_put(S + dS, K, T, r, sigma, n_steps, q)
                price_down = self.american_put(S - dS, K, T, r, sigma, n_steps, q)
        else:
            if option_type.lower() == 'call':
                price_up = self.european_call(S + dS, K, T, r, sigma, n_steps, q)
                price_down = self.european_call(S - dS, K, T, r, sigma, n_steps, q)
            else:
                price_up = self.european_put(S + dS, K, T, r, sigma, n_steps, q)
                price_down = self.european_put(S - dS, K, T, r, sigma, n_steps, q)

        delta = (price_up - price_down) / (2 * dS)

        # Gamma (second-order price sensitivity)
        gamma = (price_up - 2 * base_price + price_down) / (dS ** 2)

        # Vega (volatility sensitivity)
        d_sigma = sigma * 0.01
        if style.lower() == 'american':
            if option_type.lower() == 'call':
                price_vol_up = self.american_call(S, K, T, r, sigma + d_sigma, n_steps, q)
                price_vol_down = self.american_call(S, K, T, r, sigma - d_sigma, n_steps, q)
            else:
                price_vol_up = self.american_put(S, K, T, r, sigma + d_sigma, n_steps, q)
                price_vol_down = self.american_put(S, K, T, r, sigma - d_sigma, n_steps, q)
        else:
            if option_type.lower() == 'call':
                price_vol_up = self.european_call(S, K, T, r, sigma + d_sigma, n_steps, q)
                price_vol_down = self.european_call(S, K, T, r, sigma - d_sigma, n_steps, q)
            else:
                price_vol_up = self.european_put(S, K, T, r, sigma + d_sigma, n_steps, q)
                price_vol_down = self.european_put(S, K, T, r, sigma - d_sigma, n_steps, q)

        vega = (price_vol_up - price_vol_down) / (2 * d_sigma)

        # Theta (time sensitivity)
        dT = T * 0.01
        if T > dT:
            if style.lower() == 'american':
                if option_type.lower() == 'call':
                    price_time = self.american_call(S, K, T - dT, r, sigma, n_steps, q)
                else:
                    price_time = self.american_put(S, K, T - dT, r, sigma, n_steps, q)
            else:
                if option_type.lower() == 'call':
                    price_time = self.european_call(S, K, T - dT, r, sigma, n_steps, q)
                else:
                    price_time = self.european_put(S, K, T - dT, r, sigma, n_steps, q)

            theta = (price_time - base_price) / dT
        else:
            theta = 0.0

        # Rho (interest rate sensitivity)
        dr = r * 0.01
        if style.lower() == 'american':
            if option_type.lower() == 'call':
                price_r_up = self.american_call(S, K, T, r + dr, sigma, n_steps, q)
                price_r_down = self.american_call(S, K, T, r - dr, sigma, n_steps, q)
            else:
                price_r_up = self.american_put(S, K, T, r + dr, sigma, n_steps, q)
                price_r_down = self.american_put(S, K, T, r - dr, sigma, n_steps, q)
        else:
            if option_type.lower() == 'call':
                price_r_up = self.european_call(S, K, T, r + dr, sigma, n_steps, q)
                price_r_down = self.european_call(S, K, T, r - dr, sigma, n_steps, q)
            else:
                price_r_up = self.european_put(S, K, T, r + dr, sigma, n_steps, q)
                price_r_down = self.european_put(S, K, T, r - dr, sigma, n_steps, q)

        rho = (price_r_up - price_r_down) / (2 * dr)

        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

    def check_early_exercise_boundary(self, S: float, K: float, T: float,
                                     r: float, sigma: float, n_steps: int = 100,
                                     q: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate early exercise boundary for American options.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free interest rate
            sigma: Volatility
            n_steps: Number of time steps
            q: Dividend yield

        Returns:
            Tuple of (times, critical_stock_prices) for early exercise
        """
        dt, u, d, p = self.calculate_tree_parameters(S, K, T, r, sigma, n_steps, q)
        discount = np.exp(-r * dt)

        # Initialize stock prices and option values
        stock_prices = np.zeros((n_steps + 1, n_steps + 1))
        put_values = np.zeros((n_steps + 1, n_steps + 1))
        call_values = np.zeros((n_steps + 1, n_steps + 1))

        # Calculate stock prices at each node
        for i in range(n_steps + 1):
            for j in range(i + 1):
                stock_prices[j, i] = S * (u ** j) * (d ** (i - j))

        # Initialize option values at expiration
        for i in range(n_steps + 1):
            put_values[i, n_steps] = max(K - stock_prices[i, n_steps], 0)
            call_values[i, n_steps] = max(stock_prices[i, n_steps] - K, 0)

        # Backward induction to find early exercise boundary
        put_boundary = []
        call_boundary = []

        for step in range(n_steps - 1, -1, -1):
            put_critical = None
            call_critical = None

            for i in range(step + 1):
                # Put option
                hold_put = discount * (
                    p * put_values[i + 1, step + 1] +
                    (1 - p) * put_values[i, step + 1]
                )
                exercise_put = max(K - stock_prices[i, step], 0)
                put_values[i, step] = max(hold_put, exercise_put)

                if exercise_put > hold_put and put_critical is None:
                    put_critical = stock_prices[i, step]

                # Call option
                hold_call = discount * (
                    p * call_values[i + 1, step + 1] +
                    (1 - p) * call_values[i, step + 1]
                )
                exercise_call = max(stock_prices[i, step] - K, 0)
                call_values[i, step] = max(hold_call, exercise_call)

                if exercise_call > hold_call and call_critical is None:
                    call_critical = stock_prices[i, step]

            put_boundary.append(put_critical)
            call_boundary.append(call_critical)

        times = np.arange(n_steps) * dt

        return times, np.array(put_boundary), np.array(call_boundary)

    def converge_price(self, S: float, K: float, T: float, r: float,
                      sigma: float, option_type: str = 'call',
                      style: str = 'american', q: float = 0.0,
                      max_steps: int = 1000) -> Tuple[float, int]:
        """
        Calculate converged price using increasing number of steps.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free interest rate
            sigma: Volatility
            option_type: 'call' or 'put'
            style: 'american' or 'european'
            q: Dividend yield
            max_steps: Maximum number of steps to try

        Returns:
            Tuple of (converged_price, steps_used)
        """
        prev_price = 0.0
        steps = 10

        while steps <= max_steps:
            if style.lower() == 'american':
                if option_type.lower() == 'call':
                    current_price = self.american_call(S, K, T, r, sigma, steps, q)
                else:
                    current_price = self.american_put(S, K, T, r, sigma, steps, q)
            else:
                if option_type.lower() == 'call':
                    current_price = self.european_call(S, K, T, r, sigma, steps, q)
                else:
                    current_price = self.european_put(S, K, T, r, sigma, steps, q)

            if abs(current_price - prev_price) < self.convergence_threshold:
                break

            prev_price = current_price
            steps *= 2

        return current_price, steps


# Example usage and testing
if __name__ == "__main__":
    # Initialize binomial tree
    bt = BinomialTree()

    # Example parameters
    S = 100.0  # Stock price
    K = 105.0  # Strike price
    T = 0.25   # 3 months to expiration
    r = 0.05   # 5% risk-free rate
    sigma = 0.25  # 25% volatility
    n_steps = 100

    # Calculate prices
    euro_call = bt.european_call(S, K, T, r, sigma, n_steps)
    amer_call = bt.american_call(S, K, T, r, sigma, n_steps)
    euro_put = bt.european_put(S, K, T, r, sigma, n_steps)
    amer_put = bt.american_put(S, K, T, r, sigma, n_steps)

    print("Binomial Tree Results:")
    print(f"European Call: ${euro_call:.4f}")
    print(f"American Call: ${amer_call:.4f}")
    print(f"European Put: ${euro_put:.4f}")
    print(f"American Put: ${amer_put:.4f}")

    # Calculate Greeks
    greeks = bt.calculate_greeks(S, K, T, r, sigma, n_steps, 'put', 'american')
    print("\nAmerican Put Greeks:")
    for greek, value in greeks.items():
        print(f"{greek}: {value:.4f}")

    # Test convergence
    converged_price, steps_used = bt.converge_price(S, K, T, r, sigma, 'put', 'american')
    print(f"\nConverged Price: ${converged_price:.4f} (using {steps_used} steps)")