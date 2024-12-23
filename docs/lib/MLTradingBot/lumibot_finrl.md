Using Lumibot with FinRL involves combining the algorithmic trading capabilities of the Lumibot framework with the deep reinforcement learning (DRL) trading models provided by FinRL. Let’s walk through an overview of how to integrate the two:

1. **Set up your environment**: You’ll need Python, along with both the Lumibot and FinRL libraries installed.

   ```bash
   pip install finrl lumibot
   ```

2. **Basic structure of Lumibot**: Lumibot is designed to facilitate algorithmic trading, supporting many broker APIs (like Alpaca, Interactive Brokers, and OANDA). You’ll configure your broker account details in Lumibot, then define a trading strategy class.

3. **Implement FinRL Models**: FinRL provides pre-trained DRL agents and utilities to train custom agents. You can choose a model (such as DDPG, PPO, A2C) and train it on historical market data. Save this trained model for live trading with Lumibot.

4. **Integrate the FinRL-trained model with Lumibot**: Once you’ve trained or selected a FinRL model, the next step is to integrate it into Lumibot. 

### Detailed Steps:

#### 1. Training a Model in FinRL
To start with FinRL, train a reinforcement learning model on historical stock data. Here’s an outline of training with a Proximal Policy Optimization (PPO) model:

```python
from finrl.config import config
from finrl.model.models import DRLAgent
from finrl.env.env_stocktrading import StockTradingEnv

# Load stock data
df = ...  # Load your stock data here

# Set up environment
env = StockTradingEnv(df)

# Initialize DRL Agent
agent = DRLAgent(env)

# Train a PPO model
ppo_model = agent.train_PPO()
ppo_model.save("trained_ppo_model")  # Save the trained model
```

#### 2. Configuring Lumibot
In Lumibot, create a custom trading strategy class that uses this pre-trained FinRL model to make buy/sell decisions. 

#### 3. Defining the Strategy in Lumibot
Below is a basic setup for a Lumibot strategy that utilizes the trained model to make trading decisions.

```python
from lumibot.strategies.strategy import Strategy
from stable_baselines3 import PPO  # Import the model from FinRL's saved PPO model

class FinRLStrategy(Strategy):
    def initialize(self):
        # Load the pre-trained model
        self.model = PPO.load("trained_ppo_model")

    def on_trading_iteration(self):
        observation = self.get_observation()  # Get current market data
        action, _states = self.model.predict(observation)  # Use model to predict action
        
        if action == 1:  # Assuming 1 indicates a "Buy" action
            self.buy(symbol="AAPL", quantity=1)  # Example of placing a buy order
        elif action == 0:  # Assuming 0 indicates a "Sell" action
            self.sell(symbol="AAPL", quantity=1)  # Example of placing a sell order
```

### 4. Running the Strategy
After defining the strategy, run it on Lumibot by connecting to a broker and executing the trading loop.

```python
from lumibot.brokers.paper_trading import PaperTrading

broker = PaperTrading()  # or another broker like Alpaca or IBKR
strategy = FinRLStrategy(broker)
strategy.run()
```

### Additional Notes
- **Observation Preprocessing**: Ensure the observation format in Lumibot matches what the model expects.
- **Real-time Data**: Lumibot will pull real-time data; thus, ensure your FinRL model handles streaming data without retraining.
- **Brokerage Integration**: Make sure the broker used in Lumibot supports the assets the FinRL model was trained on.

This setup allows you to use FinRL-trained models in a live trading environment provided by Lumibot, leveraging DRL-driven decision-making in the market.