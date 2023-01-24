from tinyec import registry
from tinyec.ec import SubGroup, Curve, Point


class ECC_Crypto:
    def __init__(self) -> None:
        self.curve = registry.get_curve('secp192r1')
