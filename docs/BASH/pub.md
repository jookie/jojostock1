source venv/bin/activate


# Suppress Streamlit thread context warnings
def suppress_st_warnings():
    original_st_write = st.write
    def modified_st_write(*args, **kwargs):
        with contextlib.redirect_stdout(sys.stdout):
            with contextlib.redirect_stderr(sys.stderr):
                original_st_write(*args, **kwargs)
    st.write = modified_st_write

# Set dummy credentials
os.environ["APCA_API_KEY_ID"] = "BACKTEST"
os.environ["APCA_API_SECRET_KEY"] = "BACKTEST"

class BacktestStrategy(Strategy):
    def initialize(self):
        self.sleeptime = "1D"
        self.backtest_progress = 0
        
    def on_trading_iteration(self):
        if self.backtest_running:
            days_total = (self.datetime_end - self.datetime_start).days
            days_done = (self.get_datetime() - self.datetime_start).days
            self.backtest_progress = days_done / days_total
            st.session_state.backtest_progress = self.backtest_progress

# Initialize session state
if 'backtest_running' not in st.session_state:
    st.session_state.backtest_running = False
if 'backtest_progress' not in st.session_state:
    st.session_state.backtest_progress = 0
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

def run_backtest():
    suppress_st_warnings()  # Apply warning suppression
    try:
        broker = Alpaca(
            {"API_KEY": "BACKTEST", "API_SECRET": "BACKTEST"},
            connect_stream=False,
            backtesting=True
        )
        
        backtesting = YahooDataBacktesting(
            datetime_start=datetime(2023, 1, 1),
            datetime_end=datetime(2023, 3, 31),
            broker=broker
        )
        
        strategy = BacktestStrategy(broker=broker, budget=100000)
        strategy.backtest_running = True
        strategy.datetime_start = backtesting.datetime_start
        strategy.datetime_end = backtesting.datetime_end
        
        results = strategy.backtest()
        st.session_state.backtest_results = results
    except Exception as e:
        st.session_state.backtest_results = f"Error: {str(e)}"
    finally:
        st.session_state.backtest_running = False

def start_background_backtest():
    st.session_state.backtest_running = True
    st.session_state.backtest_results = None
    thread = threading.Thread(target=run_backtest, daemon=True)
    thread.start()

def cancel_backtest():
    st.session_state.backtest_running = False

# Streamlit UI
st.title("Backtesting Controller")

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Backtest", disabled=st.session_state.backtest_running):
        start_background_backtest()

with col2:
    if st.button("Cancel Backtest", disabled=not st.session_state.backtest_running):
        cancel_backtest()

# Progress bar
if st.session_state.backtest_running:
    progress_bar = st.progress(st.session_state.backtest_progress)
    st.caption("Backtest in progress...")
else:
    st.session_state.backtest_progress = 0

# Display results
if st.session_state.backtest_results and not st.session_state.backtest_running:
    if isinstance(st.session_state.backtest_results, str):
        st.error(st.session_state.backtest_results)
    else:
        st.success("Backtest completed successfully!")
        st.write(f"Final Portfolio Value: ${st.session_state.backtest_results['ending_portfolio_value']:,.2f}")