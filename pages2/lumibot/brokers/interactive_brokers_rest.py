import logging
from termcolor import colored
from lumibot.brokers import Broker
from lumibot.entities import Order, Asset, Position
from lumibot.data_sources import InteractiveBrokersRESTData
import datetime
from decimal import Decimal
from math import gcd
import re

TYPE_MAP = dict(
    stock="STK",
    option="OPT",
    future="FUT",
    forex="CASH",
    index="IND",
    multileg="BAG",
)

DATE_MAP = dict(
    future="%Y%m%d",
    option="%Y%m%d",
)

ORDERTYPE_MAPPING = dict(
    market="MKT",
    limit="LMT",
    stop="STP",
    stop_limit="STP LMT",
    trailing_stop="TRAIL",
)

SPREAD_CONID_MAP = {
    "AUD": 61227077,
    "CAD": 61227082,
    "CHF": 61227087,
    "CNH": 136000441,
    "GBP": 58666491,
    "HKD": 61227072,
    "INR": 136000444,
    "JPY": 61227069,
    "KRW": 136000424,
    "MXN": 136000449,
    "SEK": 136000429,
    "SGD": 426116555,
    "USD": 28812380,
}

ASSET_CLASS_MAPPING = {
    "STK": Asset.AssetType.STOCK,
    "OPT": Asset.AssetType.OPTION,
    "FUT": Asset.AssetType.FUTURE,
    "CASH": Asset.AssetType.FOREX,
}


class InteractiveBrokersREST(Broker):
    """
    Broker that connects to the Interactive Brokers REST API.
    """

    NAME = "InteractiveBrokersREST"

    def __init__(self, config, data_source=None):
        if data_source is None:
            data_source = InteractiveBrokersRESTData(config)
        super().__init__(name=self.NAME, data_source=data_source, config=config)

        self.market = "NYSE"  # The default market is NYSE.

    # --------------------------------------------------------------
    # Broker methods
    # --------------------------------------------------------------

    # Existing method stubs with logging
    def _get_balances_at_broker(self, quote_asset: Asset, strategy) -> tuple:
        """
        Get the account balances for the quote asset from the broker.

        Parameters
        ----------
        quote_asset : Asset
            The quote asset for which to retrieve the account balances.

        Returns
        -------
        tuple of float
            A tuple containing (cash, positions_value, total_liquidation_value).
            Cash = cash in the account (whatever the quote asset is).
            Positions value = the value of all the positions in the account.
            Portfolio value = the total equity value of the account (aka. portfolio value).
        """
        strategy_name = strategy._name
        # Get the account balances from the Interactive Brokers Client Portal
        account_balances = self.data_source.get_account_balances()

        # Check that the account balances were successfully retrieved
        if account_balances is None:
            logging.error(colored("Failed to retrieve account balances.", "red"))
            return 0.0, 0.0, 0.0

        # Get the quote asset symbol
        quote_symbol = quote_asset.symbol

        # account_balances = {'CHF': {'commoditymarketvalue': 0.0, 'futuremarketvalue': 0.0, 'settledcash': 188.59, 'exchangerate': 1.1847296, 'sessionid': 1, 'cashbalance': 188.59, 'corporatebondsmarketvalue': 0.0, 'warrantsmarketvalue': 0.0, 'netliquidationvalue': 188.59, 'interest': 0, 'unrealizedpnl': 0.0, 'stockmarketvalue': 0.0, 'moneyfunds': 0.0, 'currency': 'CHF', 'realizedpnl': 0.0, 'funds': 0.0, 'acctcode': 'DU4299039', 'issueroptionsmarketvalue': 0.0, 'key': 'LedgerList', ...}, 'JPY': {'commoditymarketvalue': 0.0, 'futuremarketvalue': 0.0, 'settledcash': -3794999.0, 'exchangerate': 0.0069919, 'sessionid': 1, 'cashbalance': -3794999.0, 'corporatebondsmarketvalue': 0.0, 'warrantsmarketvalue': 0.0, 'netliquidationvalue': -3794999.0, 'interest': 0, 'unrealizedpnl': 0.0, 'stockmarketvalue': 0.0, 'moneyfunds': 0.0, 'currency': 'JPY', 'realizedpnl': 0.0, 'funds': 0.0, 'acctcode': 'DU4299039', 'issueroptionsmarketvalue': 0.0, 'key': 'LedgerList', ...}, 'EUR': {'commoditymarketvalue': 0.0, 'futuremarketvalue': 0.0, 'settledcash': 287480.9, 'exchangerate': 1.1157291, 'sessionid': 1, 'cashbalance': 287480.9, 'corporatebondsmarketvalue': 0.0, 'warrantsmarketvalue': 0.0, 'netliquidationvalue': 288112.94, 'interest': 632.03, 'unrealizedpnl': 0.0, 'stockmarketvalue': 0.0, 'moneyfunds': 0.0, 'currency': 'EUR', 'realizedpnl': 0.0, 'funds': 0.0, 'acctcode': 'DU4299039', 'issueroptionsmarketvalue': 0.0, 'key': 'LedgerList', ...}, 'USD': {'commoditymarketvalue': 0.0, 'futuremarketvalue': -87.3, 'settledcash': 208917.02, 'exchangerate': 1, 'sessionid': 1, 'cashbalance': 208917.02, 'corporatebondsmarketvalue': 0.0, 'warrantsmarketvalue': 0.0, 'netliquidationvalue': 209711.64, 'interest': 518.04, 'unrealizedpnl': 19358.56, 'stockmarketvalue': 276.58, 'moneyfunds': 0.0, 'currency': 'USD', 'realizedpnl': 0.0, 'funds': 0.0, 'acctcode': 'DU4299039', 'issueroptionsmarketvalue': 0.0, 'key': 'LedgerList', ...}, 'BASE': {'commoditymarketvalue': 0.0, 'futuremarketvalue': -87.3, 'settledcash': 503393.47, 'exchangerate': 1, 'sessionid': 1, 'cashbalance': 503393.47, 'corporatebondsmarketvalue': 0.0, 'warrantsmarketvalue': 0.0, 'netliquidationvalue': 504893.34, 'interest': 1223.307, 'unrealizedpnl': 19358.56, 'stockmarketvalue': 276.58, 'moneyfunds': 0.0, 'currency': 'BASE', 'realizedpnl': 0.0, 'funds': 0.0, 'acctcode': 'DU4299039', 'issueroptionsmarketvalue': 0.0, 'key': 'LedgerList', ...}}

        # Loop through the account balances and find the quote asset. If not the quote asset, create a position object for the currency/forex asset.
        cash = 0
        balances_for_quote_asset = None
        for currency, balances in account_balances.items():
            if currency == quote_symbol:
                # Get the account balances for the quote asset
                balances_for_quote_asset = account_balances[quote_symbol]

                # Get the cash balance for the quote asset
                cash = balances_for_quote_asset["cashbalance"]
            elif currency != "BASE":
                # Create a position object for the currency/forex asset
                asset = Asset(symbol=currency, asset_type=Asset.AssetType.FOREX)
                quantity = balances["cashbalance"]

                if quantity != 0:
                    position = Position(
                        strategy=strategy_name,
                        asset=asset,
                        quantity=quantity,
                    )
                    self._filled_positions.append(position)

        # Exmaple account balances response:
        # {'commoditymarketvalue': 0.0, 'futuremarketvalue': 677.49, 'settledcash': 202142.17, 'exchangerate': 1, 'sessionid': 1, 'cashbalance': 202142.17, 'corporatebondsmarketvalue': 0.0, 'warrantsmarketvalue': 0.0, 'netliquidationvalue': 202464.67, 'interest': 452.9, 'unrealizedpnl': 12841.38, 'stockmarketvalue': -130.4, 'moneyfunds': 0.0, 'currency': 'USD', 'realizedpnl': 0.0, 'funds': 0.0, 'acctcode': 'DU4299039', 'issueroptionsmarketvalue': 0.0, 'key': 'LedgerList', 'timestamp': 1724382002, 'severity': 0, 'stockoptionmarketvalue': 0.0, 'futuresonlypnl': 677.49, 'tbondsmarketvalue': 0.0, 'futureoptionmarketvalue': 0.0, 'cashbalancefxsegment': 0.0, 'secondkey': 'USD', 'tbillsmarketvalue': 0.0, 'endofbundle': 1, 'dividends': 0.0, 'cryptocurrencyvalue': 0.0}

        # Get the net liquidation value for the quote asset
        total_liquidation_value = (
            balances_for_quote_asset["netliquidationvalue"]
            if balances_for_quote_asset is not None
            else 0
        )

        # Calculate the positions value
        positions_value = (
            (total_liquidation_value - cash) if total_liquidation_value != 0 else 0
        )

        # Check if there is a forex asset with more than 0 quantity
        if not hasattr(self, "_quote_asset_checked"):
            forex_assets_with_quantity = [
                position
                for position in self._filled_positions
                if position.asset.asset_type == Asset.AssetType.FOREX
                and position.quantity > 0
            ]

            # Recommend changing quote asset if yes
            if cash == 0 and forex_assets_with_quantity:
                logging.warning(
                    colored(
                        f"The selected quote asset '{quote_asset.symbol}' has a quantity of 0. "
                        f"Consider using a different quote asset",
                        "yellow",
                    )
                )
                self._quote_asset_checked = True

        return cash, positions_value, total_liquidation_value

    def _parse_broker_order(self, response, strategy_name, strategy_object=None):
        """Parse a broker order representation
        to an order object"""

        asset_type = [k for k, v in TYPE_MAP.items() if v == response["secType"]][0]
        totalQuantity = response["totalSize"]

        if asset_type == "multileg":
            # Create a multileg order.
            order = Order(strategy_name)
            order.order_class = Order.OrderClass.MULTILEG
            order.child_orders = []

            # Parse the legs of the combo order.
            legs = self.decode_conidex(response["conidex"])
            for leg, ratio in legs.items():
                # Create the object with just the conId
                # TODO check if all legs using the same response is an issue; test with covered calls
                child_order = self._parse_order_object(
                    strategy_name=strategy_name,
                    response=response,
                    quantity=float(ratio) * totalQuantity,
                    conId=leg,
                )
                order.child_orders.append(child_order)

        else:
            order = self._parse_order_object(
                strategy_name=strategy_name,
                response=response,
                quantity=float(totalQuantity),
                conId=response["conid"],
            )

        order._transmitted = True
        order.set_identifier(response["orderId"])
        order.status = (response["status"],)
        order.update_raw(response)
        return order

    def _parse_order_object(self, strategy_name, response, quantity, conId):
        if quantity < 0:
            side = "SELL"
            quantity = -quantity
        else:
            side = "BUY"

        symbol = response["ticker"]
        currency = response["cashCcy"]
        time_in_force = response["timeInForce"]
        limit_price = (
            response["price"]
            if "price" in response and response["price"] != ""
            else None
        )
        stop_price = (
            response["stop_price"]
            if "stop_price" in response and response["stop_price"] != ""
            else None
        )
        good_till_date = (
            response["goodTillDate"]
            if "goodTillDate" in response and response["goodTillDate"] != ""
            else None
        )

        contract_details = self.data_source.get_contract_details(conId)
        if contract_details is None:
            contract_details = {}

        secType = ASSET_CLASS_MAPPING[contract_details["instrument_type"]]

        multiplier = 1
        right = None
        strike = None
        expiration = None

        if secType == "option":
            right = contract_details["right"]
            strike = float(contract_details["strike"])

        if secType in ["option", "future"]:
            multiplier = contract_details["multiplier"]
            maturity_date = contract_details["maturity_date"]  # in YYYYMMDD

            # Format the datetime object as a string that matches the format in DATE_MAP[secType]
            expiration = datetime.datetime.strptime(maturity_date, DATE_MAP[secType])

        asset = Asset(symbol=symbol, asset_type=secType, multiplier=multiplier)

        if expiration is not None:
            asset.expiration = expiration
        if strike is not None:
            asset.strike = strike
        if right is not None:
            asset.right = right

        order = Order(
            strategy_name,
            asset,
            quantity=Decimal(quantity),
            side=side.lower(),
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            good_till_date=good_till_date,
            quote=Asset(symbol=currency, asset_type="forex"),
        )

        return order

    def _pull_broker_all_orders(self):
        """Get the broker open orders"""
        orders = self.data_source.get_open_orders()
        return orders

    def _pull_broker_order(self, identifier: str) -> Order:
        """Get a broker order representation by its id"""
        pull_order = [
            order
            for order in self.data_source.get_open_orders()
            if order.orderId == identifier
        ]
        response = pull_order[0] if len(pull_order) > 0 else None
        if response is None:
            logging.error(
                colored(f"Order with identifier {identifier} not found.", "red")
            )
            return Order(self._strategy_name)
        return response

    def _parse_broker_position(self, broker_position, strategy, orders=None):
        """Parse a broker position representation
        into a position object"""
        if broker_position["asset_type"] == "stock":
            asset = Asset(
                symbol=broker_position["symbol"],
            )
        elif broker_position["asset_type"] == "future":
            asset = Asset(
                symbol=broker_position["symbol"],
                asset_type="future",
                expiration=broker_position["expiration"],
                multiplier=broker_position["multiplier"],
            )
        elif broker_position["asset_type"] == "option":
            asset = Asset(
                symbol=broker_position["symbol"],
                asset_type="option",
                expiration=broker_position["expiration"],
                strike=broker_position["strike"],
                right=broker_position["right"],
                multiplier=broker_position["multiplier"],
            )
        elif broker_position["asset_type"] == "forex":
            asset = Asset(
                symbol=broker_position["symbol"],
                asset_type="forex",
            )
        else:  # Unreachable code.
            logging.error(
                colored(
                    f"From Interactive Brokers, asset type can only be `stock`, "
                    f"`future`, or `option`. A value of {broker_position['asset_type']} "
                    f"was received.",
                    "red",
                )
            )

        quantity = broker_position["position"]
        position = Position(strategy, asset, quantity, orders=orders)
        return position

    def _parse_broker_positions(self, broker_positions, strategy):
        """parse a list of broker positions into a
        list of position objects"""
        result = []
        for broker_position in broker_positions:
            result.append(self._parse_broker_position(broker_position, strategy))

        return result

    def _pull_position(self, strategy, asset: Asset) -> Position:
        response = self._pull_broker_positions(strategy)
        result = self._parse_broker_positions(response, strategy.name)
        for pos in result:
            if pos.asset == asset:
                return pos
        return Position(strategy, asset, 0)

    def _pull_broker_positions(self, strategy=None):
        """Get the broker representation of all positions"""
        positions = []
        ib_positions = self.data_source.get_positions()
        if ib_positions:
            for position in ib_positions:
                if position["position"] != 0:
                    positions.append(position)
        else:
            logging.debug("No positions found at interactive brokers.")

        return positions

    def _pull_positions(self, strategy) -> list[Position]:
        """
        Get the positions from the broker for the given strategy.

        Parameters
        ----------
        strategy : Strategy
            The strategy for which to retrieve the positions.

        Returns
        -------
        list of Position
            A list of Position objects representing the positions in the account.
        """

        # Get the positions from the Interactive Brokers Client Portal
        positions = self.data_source.get_positions()

        # Check that the positions were successfully retrieved
        if positions is None:
            logging.error(colored("Failed to retrieve positions.", "red"))
            return []

        # Example positions response:
        # [{'acctId': 'DU4299039', 'conid': 265598, 'contractDesc': 'AAPL', 'position': -10.0, 'mktPrice': 225.0299988, 'mktValue': -2250.3, 'currency': 'USD', 'avgCost': 211.96394, 'avgPrice': 211.96394, 'realizedPnl': 0.0, 'unrealizedPnl': -130.66, 'exchs': None, 'expiry': None, 'putOrCall': None, 'multiplier': None, 'strike': 0.0, 'exerciseStyle': None, 'conExchMap': [], 'assetClass': 'STK', 'undConid': 0}]

        # Initialize a list to store the Position objects
        positions_list = []

        # Loop through the positions and create Position objects
        for position in positions:
            # Create the Asset object for the position
            symbol = position["contractDesc"]
            if symbol.startswith("C "):
                symbol = symbol[1:].replace(" ", "")
            asset_class = ASSET_CLASS_MAPPING[position["assetClass"]]

            # If asset class is stock, create a stock asset
            if asset_class == Asset.AssetType.STOCK:
                asset = Asset(symbol=symbol, asset_type=asset_class)
            elif asset_class == Asset.AssetType.OPTION:
                # Example contract_desc: 'SPY    NOV2024 562 P [SPY   241105P00562000 100]'
                # This example format includes:
                #   - An underlying symbol at the beginning (e.g., "SPY")
                #   - Expiry and strike in human-readable format (e.g., "NOV2024 562 P")
                #   - Option details within square brackets (e.g., "[SPY   241105P00562000 100]"),
                #     where "241105P00562000" holds the expiry (YYMMDD), option type (C/P), and strike price
                contract_details = self.data_source.get_contract_details(position['conid'])

                contract_desc = position.get("contractDesc", "").strip()

                if not contract_desc:
                    logging.error("Empty contract description for option. Skipping this position.")
                    continue  # Skip processing this position as contract_desc is missing

                try:
                    # Locate the square brackets and extract the option details part
                    start_idx = contract_desc.find('[')
                    end_idx = contract_desc.find(']', start_idx)
                    
                    if start_idx == -1 or end_idx == -1:
                        logging.error(f"Brackets not found in contract description '{contract_desc}'. Expected format like '[SPY   241105P00562000 100]'.")
                        continue  # Skip if brackets are missing

                    # Extract content within brackets and find the critical pattern (e.g., "241105P00562000")
                    bracket_content = contract_desc[start_idx + 1:end_idx].strip()
                    # Search for 6 digits, followed by 'C' or 'P', followed by 8 digits for strike
                    details_match = re.search(r'\d{6}[CP]\d{8}', bracket_content)
                    
                    if not details_match:
                        logging.error(f"Expected option pattern not found in contract '{contract_desc}'.")
                        continue  # Skip if pattern does not match

                    contract_details = details_match.group(0)
                    
                    # Parse components from the details
                    expiry_raw = contract_details[:6]      # First six digits (YYMMDD format)
                    right_raw = contract_details[6]        # Seventh character (C or P)
                    strike_raw = contract_details[7:]      # Remaining characters (strike price)

                    # Check if expiry is in the correct format and convert to date
                    try:
                        expiry = datetime.datetime.strptime(expiry_raw, "%y%m%d").date()
                    except ValueError as ve:
                        logging.error(f"Invalid expiry format '{expiry_raw}' in contract '{contract_desc}': {ve}")
                        continue  # Skip this position due to invalid expiry format

                    # Convert strike to a float, assuming it’s in thousandths (e.g., "00562000" to "562.00")
                    try:
                        strike = round(float(strike_raw) / 1000, 2)
                    except ValueError as ve:
                        logging.error(f"Invalid strike price '{strike_raw}' in contract '{contract_desc}': {ve}")
                        continue  # Skip this position due to invalid strike price

                    # Validate the option type (right) as either C or P
                    if right_raw.upper() not in ["C", "P"]:
                        logging.error(f"Invalid option type '{right_raw}' in contract '{contract_desc}'. Expected 'C' or 'P'.")
                        continue  # Skip if option type is not valid

                    # Determine the option right type
                    right = Asset.OptionRight.CALL if right_raw.upper() == "C" else Asset.OptionRight.PUT

                    # Extract the underlying symbol, assumed to be the first word in contract_desc
                    underlying_asset_raw = contract_desc.split()[0]
                    
                    # Ensure underlying symbol is alphanumeric and non-empty
                    if not underlying_asset_raw.isalnum():
                        logging.error(f"Invalid underlying asset symbol '{underlying_asset_raw}' in '{contract_desc}'.")
                        continue

                    # Create the underlying asset object
                    underlying_asset = Asset(
                        symbol=underlying_asset_raw,
                        asset_type=Asset.AssetType.STOCK
                    )

                    # Create the option asset object
                    asset = Asset(
                        symbol=symbol,
                        asset_type=asset_class,
                        expiration=expiry,
                        strike=strike,
                        right=right,
                        underlying_asset=underlying_asset,
                    )

                except Exception as e:
                    logging.error(f"Error processing contract '{contract_desc}': {e}")
                    
            elif asset_class == Asset.AssetType.FUTURE:
                contract_details = self.data_source.get_contract_details(position['conid'])

                asset = Asset(
                    symbol=contract_details["symbol"],
                    asset_type=asset_class,
                    expiration=datetime.datetime.strptime(contract_details["maturity_date"], "%Y%m%d").date(),
                    multiplier=int(contract_details["multiplier"])
                )
            else:
                logging.warning(
                    colored(
                        f"Asset class '{asset_class}' not supported yet (we need to add code for this asset type): {asset_class} for position {position}",
                        "yellow",
                    )
                )
                continue

            # Create the Position object
            position_obj = Position(
                strategy=strategy,
                asset=asset,
                quantity=position["position"],
                avg_fill_price=position["avgCost"],
            )

            # Append the Position object to the list
            positions_list.append(position_obj)

        return positions_list

    def _log_order_status(self, order, status, success=True):
        if success:
            if order.order_class == Order.OrderClass.MULTILEG:
                logging.info(
                    colored(
                        "Order executed successfully: This is a multileg order.",
                        "green",
                    )
                )
                for child_order in order.child_orders:
                    logging.info(
                        colored(
                            f"Child Order: Ticker: {child_order.asset.symbol}, Quantity: {child_order.quantity}, Asset Type: {child_order.asset.asset_type}, Right: {child_order.asset.right}, Side: {child_order.side}",
                            "green",
                        )
                    )
            elif order.asset.asset_type in [
                Asset.AssetType.STOCK,
                Asset.AssetType.FOREX,
            ]:
                logging.info(
                    colored(
                        f"Order executed successfully: Ticker: {order.asset.symbol}, Quantity: {order.quantity}",
                        "green",
                    )
                )
            elif order.asset.asset_type == Asset.AssetType.OPTION:
                logging.info(
                    colored(
                        f"Order executed successfully: Ticker: {order.asset.symbol}, Expiration Date: {order.asset.expiration}, Strike: {order.asset.strike}, Right: {order.asset.right}, Quantity: {order.quantity}, Side: {order.side}",
                        "green",
                    )
                )
            elif order.asset.asset_type == Asset.AssetType.FUTURE:
                logging.info(
                    colored(
                        f"Order executed successfully: Ticker: {order.asset.symbol}, Expiration Date: {order.asset.expiration}, Multiplier: {order.asset.multiplier}, Quantity: {order.quantity}",
                        "green",
                    )
                )
            else:
                logging.info(
                    colored(
                        f"Order executed successfully: Ticker: {order.asset.symbol}, Quantity: {order.quantity}, Asset Type: {order.asset.asset_type}",
                        "green",
                    )
                )
        else:
            if order.order_class == Order.OrderClass.MULTILEG:
                logging.debug(
                    colored("Order details for failed multileg order.", "blue")
                )
                for child_order in order.child_orders:
                    logging.debug(
                        colored(
                            f"Child Order: Ticker: {child_order.asset.symbol}, Quantity: {child_order.quantity}, Asset Type: {child_order.asset.asset_type}, Right: {child_order.asset.right}, Side: {child_order.side}",
                            "blue",
                        )
                    )
            elif order.asset.asset_type in [
                Asset.AssetType.STOCK,
                Asset.AssetType.FOREX,
            ]:
                logging.debug(
                    colored(
                        f"Order details for failed {order.asset.asset_type.lower()} order: Ticker: {order.asset.symbol}, Quantity: {order.quantity}",
                        "blue",
                    )
                )
            elif order.asset.asset_type == Asset.AssetType.OPTION:
                logging.debug(
                    colored(
                        f"Order details for failed option order: Ticker: {order.asset.symbol}, Expiry Date: {order.asset.expiration}, Strike: {order.asset.strike}, Right: {order.asset.right}, Quantity: {order.quantity}, Side: {order.side}",
                        "blue",
                    )
                )
            elif order.asset.asset_type == Asset.AssetType.FUTURE:
                logging.debug(
                    colored(
                        f"Order details for failed future order: Ticker: {order.asset.symbol}, Expiry Date: {order.asset.expiration}, Multiplier: {order.asset.multiplier}, Quantity: {order.quantity}",
                        "blue",
                    )
                )
            else:
                logging.debug(
                    colored(
                        f"Order details for failed order: Ticker: {order.asset.symbol}, Quantity: {order.quantity}, Asset Type: {order.asset.asset_type}",
                        "blue",
                    )
                )

    def _submit_order(self, order: Order) -> Order:
        try:
            order_data = self.get_order_data_from_orders([order])
            response = self.data_source.execute_order(order_data)
            if response is None:
                self._log_order_status(order, "failed", success=False)
                return order
            else:
                self._log_order_status(order, "executed", success=True)

            order.identifier = response[0]["order_id"]
            order.status = "submitted"
            self._unprocessed_orders.append(order)

            return order

        except Exception as e:
            logging.error(
                colored(
                    f"An error occurred while submitting the order: {str(e)}", "red"
                )
            )
            logging.error(colored(f"Error details:", "red"), exc_info=True)
            return order

    def submit_orders(
        self,
        orders: list[Order],
        is_multileg: bool = False,
        order_type: str = "market",
        duration: str = "day",
        price=None,
    ):
        try:
            if is_multileg:
                if order_type == "credit":
                    if price is not None:
                        order_type = "limit"
                        if price < 0:
                            price = -price
                    else:
                        order_type = "market"

                elif order_type == "debit":
                    if price is not None:
                        order_type = "limit"
                    else:
                        order_type = "market"

                elif order_type == "even":
                    price = 0
                    order_type = "limit"

                order_data = self.get_order_data_multileg(
                    orders, order_type=order_type, duration=duration, price=price
                )
                response = self.data_source.execute_order(order_data)
                if response is None:
                    for order in orders:
                        self._log_order_status(order, "failed", success=False)
                    return None

                order = Order(orders[0].strategy)
                order.order_class = Order.OrderClass.MULTILEG
                order.child_orders = orders
                order.status = "submitted"
                order.identifier = response[0]["order_id"]

                self._unprocessed_orders.append(order)
                self._log_order_status(order, "executed", success=True)
                return [order]

            else:
                order_data = self.get_order_data_from_orders([order])
                response = self.data_source.execute_order(order_data)
                if response is None:
                    for order in orders:
                        self._log_order_status(order, "failed", success=False)
                    return None

                # TODO Could be a problematic system
                order_id = 0
                for order in orders:
                    order.status = "submitted"
                    order.identifier = response[order_id]["order_id"]
                    self._unprocessed_orders.append(order)
                    self._log_order_status(order, "executed", success=True)
                    order_id += 1

                return orders

        except Exception as e:
            logging.error(
                colored(
                    f"An error occurred while submitting the order: {str(e)}", "red"
                )
            )
            logging.error(colored(f"Error details:", "red"), exc_info=True)

    def cancel_order(self, order: Order) -> None:
        self.data_source.delete_order(order)

    def decode_conidex(self, conidex: str) -> dict:
        # Decode this format {spread_conid};;;{leg_conid1}/{ratio},{leg_conid2}/{ratio}
        string = conidex
        _, ratios = string.split(";;;")
        legs = ratios.split(",")

        legs_dict = {}
        for leg in legs:
            leg_conid, ratio = leg.split("/")
            legs_dict[leg_conid] = ratio

        return legs_dict

    def get_order_data_from_order(self, order):
        try:
            conid = None
            side = None
            orderType = None

            if order.is_buy_order():
                side = "BUY"
            elif order.is_sell_order():
                side = "SELL"
            else:
                logging.error(colored("Order Side Not Found", "red"))
                return None

            orderType = ORDERTYPE_MAPPING[order.type]

            conid = self.data_source.get_conid_from_asset(order.asset)

            if conid is None:
                asset_type = order.asset.asset_type
                expiry_date = (
                    order.asset.expiration
                    if hasattr(order.asset, "expiration")
                    else "N/A"
                )
                logging.error(
                    colored(
                        f"Couldn't find an appropriate asset for {order.asset} (Type: {asset_type}, Expiry: {expiry_date}).",
                        "red",
                    )
                )
                return None

            rules = self.data_source.get_contract_rules(conid)
            increment = rules['rules']['increment'] # 0.05 for example
            price = (order.limit_price // increment) * increment if order.limit_price is not None else None
            aux_price = (order.stop_price // increment) * increment if order.stop_price is not None else None

            data = {
                "conid": conid,
                "quantity": round(order.quantity, 2),
                "orderType": orderType,
                "side": side,
                "tif": order.time_in_force.upper(),
                "price": price,
                "auxPrice": aux_price,
                "listingExchange": order.exchange,
            }

            if order.trail_percent:
                data["trailingType"] = "%"
                data["trailingAmt"] = order.trail_percent

            if order.trail_price:
                data["trailingType"] = "amt"
                data["trailingAmt"] = order.trail_price

            # Remove items with value None from order_data
            data = {k: v for k, v in data.items() if v is not None}
            return data

        except Exception as e:
            logging.error(
                colored(
                    f"An error occurred while processing the order: {str(e)}", "red"
                )
            )
            logging.error(colored(f"Error details:", "red"), exc_info=True)
            return None

    def get_order_data_from_orders(self, orders: list[Order]):
        order_data = {"orders": []}

        for order in orders:
            data = self.get_order_data_from_order(order)
            if data is not None:
                order_data["orders"].append(data)

        return order_data if order_data["orders"] else None

    def get_order_data_multileg(
        self, orders: list[Order], order_type=None, duration=None, price=None
    ):
        """
        Generate the order data for a multileg order.

        Parameters
        ----------
        orders : list[Order]
            List of Order objects representing the legs of the multileg order.
        order_type : str, optional
            The type of the order (e.g., 'market', 'limit'). Defaults to None.
        duration : str, optional
            The duration of the order (e.g., 'day', 'gtc'). Defaults to None.
        price : float, optional
            The price of the order. Defaults to None.

        Returns
        -------
        dict
            A dictionary containing the order data for the multileg order.
        """

        # Initialize the order data dictionary
        order_data = {"orders": []}

        # Ensure the first order has a quote asset
        if orders[0].quote is None:
            logging.error("Quote is None for the first order.")
            return None

        # Get the spread conid for the quote asset
        spread_conid = SPREAD_CONID_MAP.get(orders[0].quote.symbol)
        if spread_conid is None:
            logging.error(colored("Spread conid Not Found", "red"))
            return None

        # Build the conidex string in the format {spread_conid};;;{leg_conid1}/{ratio},{leg_conid2}/{ratio}
        conidex = f"{spread_conid};;;"

        # List to store conid and quantity pairs
        ratios = []

        # Loop through each order to get the conid and quantity
        for order in orders:
            side = None
            conid = None

            # Determine the side of the order (buy or sell)
            if order.is_buy_order():
                side = "BUY"
            elif order.is_sell_order():
                side = "SELL"
            else:
                logging.error(colored("Order Side Not Found", "red"))
                return None

            # Get the conid for the asset
            conid = self.data_source.get_conid_from_asset(order.asset)
            if conid is None:
                logging.error(colored("Order conid Not Found", "red"))
                return None

            # Get the quantity of the order
            quantity = order.quantity
            if quantity == 0 or quantity is None:
                return None

            # If the order is a sell, make the quantity negative
            if side == "SELL":
                quantity = -quantity

            # Append the conid and quantity pair to the ratios list
            ratios.append((conid, quantity))

        # Calculate the greatest common divisor (GCD) of the quantities to simplify the conidex
        quantities = [quant for _, quant in ratios]
        order_quantity = gcd(*quantities)

        # Build the conidex string with the simplified quantities
        first_order = True
        for conid, quantity in ratios:
            if first_order:
                first_order = False
            else:
                conidex += ","
            conidex += f"{conid}/{quantity // order_quantity}"

        # Set the side to "BUY" for the multileg order
        side = "BUY"

        if not orders:
            logging.error("Orders list cannot be empty")

        order = orders[0]

        # Determine the order type, defaulting to "MKT" if not specified
        order_type_value = order_type if order_type is not None else order.type
        if order_type_value is None:
            order_type_value = "MKT"
            logging.info("Order type not specified. Defaulting to 'MKT'.")

        rules = self.data_source.get_contract_rules(conid)
        increment = rules['rules']['increment'] # 0.05 for example
        price = (price // increment) * increment if price is not None else None
        aux_price = (order.stop_price // increment) * increment if order.stop_price is not None else None

        # Build the order data dictionary
        data = {
            "conidex": conidex,
            "quantity": round(order_quantity, 2),
            "orderType": ORDERTYPE_MAPPING.get(order_type_value),
            "side": side,
            "tif": duration.upper()
            if duration is not None
            else order.time_in_force.upper(),
            "price": price,
            "auxPrice": aux_price,
            "listingExchange": order.exchange,
        }

        # Remove items with value None from the order data
        data = {k: v for k, v in data.items() if v is not None}
        order_data["orders"].append(data)

        return order_data

    def get_historical_account_value(self) -> dict:
        logging.error(
            "The function get_historical_account_value is not implemented yet for Interactive Brokers."
        )
        return {"hourly": None, "daily": None}

    def _register_stream_events(self):
        logging.error(
            colored("Method '_register_stream_events' is not yet implemented.", "red")
        )
        return None

    def _run_stream(self):
        logging.error(colored("Method '_run_stream' is not yet implemented.", "red"))
        return None

    def _get_stream_object(self):
        logging.warning(
            colored("Method '_get_stream_object' is not yet implemented.", "yellow")
        )
        return None

    def _close_connection(self):
        logging.info("Closing connection to the Client Portal...")
        self.data_source.stop()
