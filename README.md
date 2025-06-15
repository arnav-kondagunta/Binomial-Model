# Binomial-Model
Building on my Black-Scholes Model project, I built a Binomial Options Pricing Model that outputs the price of an option based on provided stock ticker and additional financial data. It uses data from Yahoo Finance to fetch stock data and the Federal Reserve Economic Data (FRED) to fetch risk-free treasury rates. Based on the inputs, the model builds a binomial tree model to map out the stock price paths and uses the option payoff function to create a second tree model of the corresponding option price paths. Finally, the price of the option is given by the discounted conditional expectation of the option price at the initial time step.
## Directed Reading Program
This project was created to supplement my Rutgers Directed Reading Program (Spring 2025) project. As part of this program, I conducted a study of discrete-time derivatives pricing, culminating in final presentation. Attached in this repo is my final project.
## Files
The Python file uses Streamlit to create an interactive web application that will take in the user inputs and provide the price of the derivative. On the other hand, the Jupyter Notebook lets the user run the code in their terminal.
