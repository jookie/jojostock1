import streamlit as st
import ast
from datetime import datetime

# Path to the config file
config_file_path = 'pages/config.py'

def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        exec(file.read(), config)
    return config

def write_config(file_path, config):
    with open(file_path, 'w') as file:
        file.write("# Configuration File\n\n")
        for key, value in config.items():
            if not key.startswith('__'):
                if isinstance(value, str):
                    file.write(f'{key} = "{value}"\n')
                elif isinstance(value, list):
                    file.write(f'{key} = {value}\n')
                elif isinstance(value, dict):
                    file.write(f'{key} = {value}\n')
                else:
                    file.write(f'{key} = {value}\n')

def main():
    st.title("Settings Page")

    # Read the configuration
    config = read_config(config_file_path)

    # Convert date strings to date objects
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return None

    # List of all possible indicators
    all_indicators = [
        "macd", "boll_ub", "boll_lb", "rsi_30", 
        "cci_30", "dx_30", "close_30_sma", "close_60_sma"
    ]

    # Create a form
    with st.form(key='settings_form'):
        # Create two columns for directory settings
        col1, col2 = st.columns(2)

        with col1:
            st.header("Directory Settings")
            data_save_dir = st.text_input("Data Save Directory", value=config.get('DATA_SAVE_DIR', ''))
            trained_model_dir = st.text_input("Trained Model Directory", value=config.get('TRAINED_MODEL_DIR', ''))

        with col2:
            tensorboard_log_dir = st.text_input("Tensorboard Log Directory", value=config.get('TENSORBOARD_LOG_DIR', ''))
            results_dir = st.text_input("Results Directory", value=config.get('RESULTS_DIR', ''))

        # Organize each date pair in a single row
        st.header("Date Settings")

        train_col1, train_col2 = st.columns(2)
        with train_col1:
            train_start_date = st.date_input(
                "Train Start Date", 
                value=parse_date(config.get('TRAIN_START_DATE', ''))
            )
        with train_col2:
            train_end_date = st.date_input(
                "Train End Date", 
                value=parse_date(config.get('TRAIN_END_DATE', ''))
            )

        test_col1, test_col2 = st.columns(2)
        with test_col1:
            test_start_date = st.date_input(
                "Test Start Date", 
                value=parse_date(config.get('TEST_START_DATE', ''))
            )
        with test_col2:
            test_end_date = st.date_input(
                "Test End Date", 
                value=parse_date(config.get('TEST_END_DATE', ''))
            )

        st.header("API Settings")
        api_col1, api_col2 = st.columns(2)
        with api_col1:
            alpaca_api_key = st.text_input("Alpaca API Key", value=config.get('ALPACA_API_KEY', ''))
        with api_col2:
            alpaca_api_secret = st.text_input("Alpaca API Secret", value=config.get('ALPACA_API_SECRET', ''))
        alpaca_api_base_url = st.text_input("Alpaca API Base URL", value=config.get('ALPACA_API_BASE_URL', ''))

        st.header("Technical Indicators")
        # Multi-selection widget for indicators
        indicators = st.multiselect(
            "Select Technical Indicators", 
            options=all_indicators, 
            default=config.get('INDICATORS', [])
        )

        # Model Parameters Section
        st.header("Model Parameters")

        def display_model_params(param_name, params):
            st.subheader(param_name)
            param_col1, param_col2 = st.columns(2)  # Two columns for each model's parameters
            param_values = {}
            with param_col1:
                for i, (key, value) in enumerate(params.items()):
                    if i % 2 == 0:  # Place every other parameter in the first column
                        if isinstance(value, int):
                            param_values[key] = st.number_input(f"{key}", value=value, step=1)
                        elif isinstance(value, float):
                            param_values[key] = st.number_input(f"{key}", value=value, format="%.6f")
                        elif isinstance(value, str):
                            param_values[key] = st.text_input(f"{key}", value=value)
            with param_col2:
                for i, (key, value) in enumerate(params.items()):
                    if i % 2 != 0:  # Place every other parameter in the second column
                        if isinstance(value, int):
                            param_values[key] = st.number_input(f"{key}", value=value, step=1)
                        elif isinstance(value, float):
                            param_values[key] = st.number_input(f"{key}", value=value, format="%.6f")
                        elif isinstance(value, str):
                            param_values[key] = st.text_input(f"{key}", value=value)
            return param_values

        # Display each model's parameters
        a2c_params = display_model_params("A2C_PARAMS", config.get('A2C_PARAMS', {}))
        ppo_params = display_model_params("PPO_PARAMS", config.get('PPO_PARAMS', {}))
        ddpg_params = display_model_params("DDPG_PARAMS", config.get('DDPG_PARAMS', {}))
        td3_params = display_model_params("TD3_PARAMS", config.get('TD3_PARAMS', {}))
        sac_params = display_model_params("SAC_PARAMS", config.get('SAC_PARAMS', {}))
        erl_params = display_model_params("ERL_PARAMS", config.get('ERL_PARAMS', {}))

        st.header("Miscellaneous Settings")
        use_timezone_selfdefined = st.checkbox("Use Self-defined Timezone", value=config.get('USE_TIME_ZONE_SELFDEFINED', 0))

        # Submit button for the form
        submit_button = st.form_submit_button(label='Save Settings')

    # If the form is submitted, update and save the config
    if submit_button:
        config['DATA_SAVE_DIR'] = data_save_dir
        config['TRAINED_MODEL_DIR'] = trained_model_dir
        config['TENSORBOARD_LOG_DIR'] = tensorboard_log_dir
        config['RESULTS_DIR'] = results_dir
        config['TRAIN_START_DATE'] = train_start_date.strftime("%Y-%m-%d")
        config['TRAIN_END_DATE'] = train_end_date.strftime("%Y-%m-%d")
        config['TEST_START_DATE'] = test_start_date.strftime("%Y-%m-%d")
        config['TEST_END_DATE'] = test_end_date.strftime("%Y-%m-%d")
        config['ALPACA_API_KEY'] = alpaca_api_key
        config['ALPACA_API_SECRET'] = alpaca_api_secret
        config['ALPACA_API_BASE_URL'] = alpaca_api_base_url
        config['INDICATORS'] = indicators
        config['A2C_PARAMS'] = a2c_params
        config['PPO_PARAMS'] = ppo_params
        config['DDPG_PARAMS'] = ddpg_params
        config['TD3_PARAMS'] = td3_params
        config['SAC_PARAMS'] = sac_params
        config['ERL_PARAMS'] = erl_params
        config['USE_TIME_ZONE_SELFDEFINED'] = int(use_timezone_selfdefined)

        write_config(config_file_path, config)
        st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()
