import pandas as pd
from sigalg.finance import BlackScholes, Claim


class TestFinanceMethods:
    def test_delta_against_hull_example(self):
        """Test the delta method against an example from Hull's book. (Table 19.2 in the 11th edition.)"""
        data = pd.DataFrame(
            [
                [
                    49.0,
                    48.12,
                    47.37,
                    50.25,
                    51.75,
                    53.12,
                    53,
                    51.87,
                    51.38,
                    53,
                    49.88,
                    48.5,
                    49.88,
                    50.37,
                    52.13,
                    51.88,
                    52.87,
                    54.87,
                    54.62,
                    55.87,
                    57.25,
                ]
            ],
            columns=[k / 52 for k in range(21)],
        )
        S = BlackScholes.from_data(
            data=data,
            risk_free_rate=0.05,
            volatility=0.2,
        )
        expected_delta = pd.DataFrame(
            [
                [
                    0.52160466,
                    0.45800967,
                    0.40002677,
                    0.59628195,
                    0.69295199,
                    0.77382062,
                    0.77128923,
                    0.70615131,
                    0.67418957,
                    0.78652638,
                    0.55016889,
                    0.41275439,
                    0.54246684,
                    0.59052225,
                    0.76820719,
                    0.75918769,
                    0.86504218,
                    0.97824837,
                    0.98987878,
                    0.99997449,
                ]
            ],
            columns=pd.Index(
                [
                    0.0,
                    0.01923077,
                    0.03846154,
                    0.05769231,
                    0.07692308,
                    0.09615385,
                    0.11538462,
                    0.13461538,
                    0.15384615,
                    0.17307692,
                    0.19230769,
                    0.21153846,
                    0.23076923,
                    0.25,
                    0.26923077,
                    0.28846154,
                    0.30769231,
                    0.32692308,
                    0.34615385,
                    0.36538462,
                ],
                name="t",
            ),
            index=pd.Index([0], name="omega"),
        )
        Phi = Claim.european_option(strike=50, model=S)
        delta = Phi.delta()

        pd.testing.assert_frame_equal(delta.data, expected_delta, check_names=False)
