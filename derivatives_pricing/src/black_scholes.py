"""
Black-Scholes-Merton Option Pricing Model

Implementation of the classic Black-Scholes-Merton model for European options
with complete Greeks calculation and implied volatility computation.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from typing import Tuple, Union
import warnings

class BlackScholesMerton:
    """
    Black-Scholes-Merton option pricing model with Greeks calculation.

    This class implements analytical solutions for European options and
    provides complete sensitivity analysis (Greeks) and implied volatility
    computation using numerical methods.
    """

    def __init__(self):
        self.sqrt_2pi = np.sqrt(2 * np.pi)

    def calculate_d1_d2(self, S: float, K: float, T: float,
                       r: float, sigma: float, q: float = 0.0) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            q: Dividend yield (default: 0)

        Returns:
            Tuple of (d1, d2) values
        """
        if T <= 0:
            return 0.0, 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return d1, d2

    def call_price(self, S: float, K: float, T: float,
                   r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate European call option price using Black-Scholes-Merton.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)

        d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma, q)

        call_price = (S * np.exp(-q * T) * norm.cdf(d1) -
                     K * np.exp(-r * T) * norm.cdf(d2))

        return call_price

    def put_price(self, S: float, K: float, T: float,
                  r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate European put option price using Black-Scholes-Merton.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)

        d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma, q)

        put_price = (K * np.exp(-r * T) * norm.cdf(-d2) -
                    S * np.exp(-q * T) * norm.cdf(-d1))

        return put_price

    def calculate_greeks(self, S: float, K: float, T: float,
                        r: float, sigma: float, q: float = 0.0) -> dict:
        """
        Calculate complete set of Greeks for European options.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Dictionary containing all Greeks
        """
        if T <= 0:
            return self._expiration_greeks(S, K)

        d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma, q)

        # Common terms
        sqrt_T = np.sqrt(T)
        exp_r_T = np.exp(-r * T)
        exp_q_T = np.exp(-q * T)
        norm_d1 = norm.cdf(d1)
        norm_d2 = norm.cdf(d2)
        norm_pdf_d1 = norm.pdf(d1)

        # Greeks calculations
        delta_call = exp_q_T * norm_d1
        delta_put = exp_q_T * (norm_d1 - 1)

        gamma = exp_q_T * norm_pdf_d1 / (S * sigma * sqrt_T)

        vega_call = S * exp_q_T * norm_pdf_d1 * sqrt_T / 100  # Vega per 1% change
        vega_put = vega_call  # Vega is same for calls and puts

        theta_call = (-S * exp_q_T * norm_pdf_d1 * sigma / (2 * sqrt_T) -
                     r * K * exp_r_T * norm_d2 +
                     q * S * exp_q_T * norm_d1) / 365  # Theta per day

        theta_put = (-S * exp_q_T * norm_pdf_d1 * sigma / (2 * sqrt_T) +
                    r * K * exp_r_T * norm_cdf(-d2) -
                    q * S * exp_q_T * norm_cdf(-d1)) / 365

        rho_call = K * T * exp_r_T * norm_d2 / 100  # Rho per 1% change
        rho_put = -K * T * exp_r_T * norm_cdf(-d2) / 100

        return {
            'delta_call': delta_call,
            'delta_put': delta_put,
            'gamma': gamma,
            'vega': vega_call,
            'theta_call': theta_call,
            'theta_put': theta_put,
            'rho_call': rho_call,
            'rho_put': rho_put,
            'd1': d1,
            'd2': d2
        }

    def _expiration_greeks(self, S: float, K: float) -> dict:
        """Calculate Greeks at expiration."""
        if S > K:
            return {
                'delta_call': 1.0,
                'delta_put': 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta_call': 0.0,
                'theta_put': 0.0,
                'rho_call': 0.0,
                'rho_put': 0.0,
                'd1': 0.0,
                'd2': 0.0
            }
        else:
            return {
                'delta_call': 0.0,
                'delta_put': -1.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta_call': 0.0,
                'theta_put': 0.0,
                'rho_call': 0.0,
                'rho_put': 0.0,
                'd1': 0.0,
                'd2': 0.0
            }

    def implied_volatility(self, market_price: float, S: float, K: float,
                          T: float, r: float, option_type: str = 'call',
                          q: float = 0.0, initial_guess: float = 0.2,
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility from market price using Newton-Raphson method.

        Args:
            market_price: Observed market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            option_type: 'call' or 'put'
            q: Dividend yield
            initial_guess: Initial volatility guess
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance

        Returns:
            Implied volatility
        """
        if T <= 0:
            return 0.0

        def price_diff(sigma):
            if option_type.lower() == 'call':
                return self.call_price(S, K, T, r, sigma, q) - market_price
            else:
                return self.put_price(S, K, T, r, sigma, q) - market_price

        def vega(sigma):
            _, d2 = self.calculate_d1_d2(S, K, T, r, sigma, q)
            d1, _ = self.calculate_d1_d2(S, K, T, r, sigma, q)
            return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        try:
            implied_vol = newton(
                price_diff,
                initial_guess,
                fprime=vega,
                maxiter=max_iterations,
                tol=tolerance
            )

            # Ensure positive volatility
            return max(implied_vol, 0.001)

        except (ValueError, RuntimeError):
            # Fallback to bisection method if Newton-Raphson fails
            return self._bisection_iv(market_price, S, K, T, r, option_type, q)

    def _bisection_iv(self, market_price: float, S: float, K: float,
                     T: float, r: float, option_type: str,
                     q: float = 0.0) -> float:
        """Fallback bisection method for implied volatility."""
        low, high = 0.001, 5.0

        for _ in range(100):
            mid = (low + high) / 2

            if option_type.lower() == 'call':
                price = self.call_price(S, K, T, r, mid, q)
            else:
                price = self.put_price(S, K, T, r, mid, q)

            diff = price - market_price

            if abs(diff) < 1e-6:
                return mid
            elif diff > 0:
                high = mid
            else:
                low = mid

        return (low + high) / 2

    def calculate_all_prices(self, S: float, strikes: np.ndarray, T: float,
                            r: float, sigma: float, q: float = 0.0) -> dict:
        """
        Calculate call and put prices for multiple strikes.

        Args:
            S: Current stock price
            strikes: Array of strike prices
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Dictionary with call and put price arrays
        """
        strikes = np.asarray(strikes)
        call_prices = np.array([
            self.call_price(S, K, T, r, sigma, q) for K in strikes
        ])
        put_prices = np.array([
            self.put_price(S, K, T, r, sigma, q) for K in strikes
        ])

        return {
            'strikes': strikes,
            'call_prices': call_prices,
            'put_prices': put_prices
        }

    def validate_parameters(self, S: float, K: float, T: float,
                           r: float, sigma: float) -> bool:
        """
        Validate input parameters for Black-Scholes calculation.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free interest rate
            sigma: Volatility

        Returns:
            True if parameters are valid
        """
        if S <= 0 or K <= 0:
            warnings.warn("Stock price and strike price must be positive")
            return False

        if T < 0:
            warnings.warn("Time to expiration cannot be negative")
            return False

        if sigma < 0:
            warnings.warn("Volatility cannot be negative")
            return False

        return True


# Example usage and testing
if __name__ == "__main__":
    # Initialize Black-Scholes calculator
    bsm = BlackScholesMerton()

    # Example parameters
    S = 100.0  # Stock price
    K = 105.0  # Strike price
    T = 0.25   # 3 months to expiration
    r = 0.05   # 5% risk-free rate
    sigma = 0.25  # 25% volatility

    # Calculate prices
    call_price = bsm.call_price(S, K, T, r, sigma)
    put_price = bsm.put_price(S, K, T, r, sigma)

    print(f"Call Price: ${call_price:.4f}")
    print(f"Put Price: ${put_price:.4f}")

    # Calculate Greeks
    greeks = bsm.calculate_greeks(S, K, T, r, sigma)
    print("\nGreeks:")
    for greek, value in greeks.items():
        print(f"{greek}: {value:.4f}")

    # Calculate implied volatility
    market_price = 5.20
    iv = bsm.implied_volatility(market_price, S, K, T, r, 'call')
    print(f"\nImplied Volatility: {iv:.4f} ({iv*100:.2f}%)")