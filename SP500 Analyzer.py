#Libraries
import pandas as pd
from datetime import datetime
import yfinance as yf
import numpy as np
from datetime import datetime
from yahooquery import Ticker
from pathlib import Path
import os

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR  = BASE_DIR / "plots"
for d in (DATA_DIR, OUTPUT_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")

filepath   = DATA_DIR   / "SP500_stocks_data.csv"
filepath_2 = DATA_DIR   / "df_analysis_data.csv"
filepath_3 = OUTPUT_DIR / f"final_results_data_{today}.csv"

#Testing Flag (0-no / 1-yes)

testing_flag = 0
testing_flag_step_IV = 0

#I Downloading SP500 tickers - done
if testing_flag == 0:
    try:
        #URL to the CSV file containing S&P 500 constituents
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        #Read the CSV file into a DataFrame
        df = pd.read_csv(url)
        #Extract the ticker symbols and replace dots with hyphens (for Yahoo Finance compatibility)
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    except:
        print("An Error occured while proceeding I step - retreiving ticker symbols...")
else:
    print("You are in the Testing Mode!")

print("Step I - Downloading tickers finished!")

#II Downloading SP500 stocks
#try to download, if downloading is not possible, then set up very high price for the company to be at the bottom of the list
#Set up dates
try:
    if testing_flag == 0:
        start_date = "2018-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        #tickers = ["AAPL", "UNH", "XXXXX"]  #unblock line for test purposes
        errors = []

        #Stwórz wspólny indeks dat
        date_index = pd.date_range(start=start_date, end=end_date, freq='B')

        #Creating a new data frame with NaN values for every ticker
        sp500_data = pd.DataFrame(np.nan, index=date_index, columns=tickers)
        # df_test = yf.download("AAPL", start=start_date, end=end_date)
        # print(df_test)

        for ticker in tickers:
            try:
                print(f"Downloading data for: {ticker}")
                df_close = yf.download(ticker, start=start_date, end=end_date)['Close']
                
                if df_close.empty:
                    raise ValueError("No data to download")
                    
                #Match with a proper column
                sp500_data[ticker] = df_close.reindex(date_index)
                
            except Exception as e:
                print(f"No data to download {ticker}: {e}")
                errors.append(ticker)

            #print(sp500_data.head())
            
        #Save the data
        print("Step II - Downloading the stocks data finished!")
        sp500_data = sp500_data.dropna(axis=0, how='all')
        print(sp500_data.head())
        try:
            sp500_data.to_csv(filepath, index=True)
        except:
            print("Couldn't save the file")
    else:
        print("You are in the Testing Mode!")
        sp500_data = pd.read_csv(filepath, index_col=0)
except:
    print("An Error occured while proceeding II step - downloading stock data...")
print(sp500_data.head())
print("II Step is finished!")

#III Calculating Drawdown -  to check

#Global Peak
global_peak = sp500_data.max()  #Series with max for every company

#Current Prices
current_prices = sp500_data.iloc[-1]  #Series with latest prices

#Current Drawdown
current_drawdown = (global_peak - current_prices) / global_peak

#Drawdown in percentages
current_drawdown_percentages = current_drawdown * 100

#Create a Data Frame - AXIS 1
df_analysis = pd.DataFrame({
    "Global Peak": global_peak,
    "Current Price": current_prices,
    "Drawdown (%)": current_drawdown_percentages
}).sort_values("Drawdown (%)", ascending=False)

print(df_analysis.head())

#IV Downloading fundamental characterictics

#Adding columns to df_analysis
df_analysis['Company Name'] = None
df_analysis['Sector'] = None
df_analysis['P/E (TTM)'] = None #Price/Earnings Ratio - lower the better (0-20 ideal) | Formula: Stock Price / Earnings Per Share
df_analysis['P/B'] = None #Price to Book Ratio - lower the better (0.5-2.5 ideal) | Formula: Market Price / Book Value per Share
df_analysis['D/E'] = None #Debt to Equity - lower the better (0-1.5 ideal) | Formula: Total Debt / Shareholders' Equity
df_analysis['ROE'] = None #Return on Equity - higher the better (15-30% ideal) | Formula: Net Income / Shareholders' Equity × 100%
df_analysis['Div Yield'] = None #Dividend Yield - higher the better (2-6% ideal) | Formula: Annual Dividends / Stock Price × 100%
df_analysis['Current Ratio'] = None #Current Assets/Current Liabilities - higher the better (1.5-3.0 ideal) | Formula: Current Assets / Current Liabilities
df_analysis['Profit Margin'] = None #Higher the better (10-25% ideal) | Formula: Net Income / Revenue × 100%

print(df_analysis.head(10))

if testing_flag_step_IV == 0:
    for ticker in df_analysis.index:
        try:
            print(f"Downloading fundamentals for: {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info
            #Add values to the data frame
            df_analysis.loc[ticker, 'Company Name'] = info.get('longName')
            df_analysis.loc[ticker, 'Sector'] = info.get('sector')
            df_analysis.loc[ticker, 'P/E (TTM)'] = info.get('trailingPE')
            df_analysis.loc[ticker, 'P/B'] = info.get('priceToBook')
            df_analysis.loc[ticker, 'D/E'] = info.get('debtToEquity')
            df_analysis.loc[ticker, 'ROE'] = info.get('returnOnEquity')
            df_analysis.loc[ticker, 'Div Yield'] = info.get('dividendYield')
            df_analysis.loc[ticker, 'Current Ratio'] = info.get('currentRatio')
            df_analysis.loc[ticker, 'Profit Margin'] = info.get('profitMargins')
            #add name of company and sector

        except:
            print(f"Error downloading fundamentals for {ticker}")
        
        df_analysis.to_csv(filepath_2, index=True)
else:
    print("You are in the Testing Mode!")
    df_analysis = pd.read_csv(filepath_2, index_col=0)

print("Fundamental data downloaded")
stocks_with_na = df_analysis.isnull().any(axis=1).sum()
print(f"Number of stocks with N/A data: {stocks_with_na}")

#Deleting stocks with missing data
df_analysis = df_analysis.dropna()
print(f"Number of stocks after dropping one's with missing data: {len(df_analysis)}")
print(df_analysis.head(10))

#Colums to check outliers
columns_to_check = ['P/E (TTM)', 'P/B', 'D/E', 'ROE', 'Div Yield', 'Current Ratio', 'Profit Margin']

#Printing results
print(f"Number of stocks after removing outliers: {len(df_analysis)}")
print(df_analysis)

#V Prepare & export ranking
#The goal of this part is to select stocks that are worth buying.
#The code will maximize the current drawdown and filter fundamental factors
#to include only those stocks that meet strict investment criteria.
print(df_analysis.loc['AAPL']) #printing one row to check

df_rank = df_analysis.copy()

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

#Normalization of maximum/minimum rates
df_rank['Drawdown_norm'] = min_max_normalize(df_rank['Drawdown (%)'])
df_rank['PE_norm'] = 1 - min_max_normalize(df_rank['P/E (TTM)'])
df_rank['PB_norm'] = 1 - min_max_normalize(df_rank['P/B'])
df_rank['DE_norm'] = 1 - min_max_normalize(df_rank['D/E'])
df_rank['ROE_norm'] = min_max_normalize(df_rank['ROE'])
df_rank['DivYield_norm'] = min_max_normalize(df_rank['Div Yield'].fillna(0))
df_rank['CurrentRatio_norm'] = min_max_normalize(df_rank['Current Ratio'])
df_rank['ProfitMargin_norm'] = min_max_normalize(df_rank['Profit Margin'])

# #ranking
df_rank['Total_score'] = (
    df_rank['Drawdown_norm'] +
    df_rank['PE_norm'] +
    df_rank['PB_norm'] +
    df_rank['DE_norm'] +
    df_rank['ROE_norm'] +
    df_rank['DivYield_norm'] +
    df_rank['CurrentRatio_norm'] +
    df_rank['ProfitMargin_norm']
) / 8 #dividing by 8 shows the average score for each stock


df_rank = df_rank.sort_values(by='Total_score', ascending = False)
df_rank['Score Rank ']=range(1, len(df_rank)+1)

criteria = [
    ('Drawdown_norm', True),       # True = the bigger value the better
    ('PE_norm', True),
    ('PB_norm', True),
    ('DE_norm', True),
    ('ROE_norm', True),
    ('DivYield_norm', True),
    ('CurrentRatio_norm', True),
    ('ProfitMargin_norm', True)
]

for col, bigger_is_better in criteria:
    df_rank[col+'_rank'] = df_rank[col].rank(method='min', ascending = not bigger_is_better)

rank_cols = [c for c in df_rank.columns if c.endswith('_rank')] #identification of rank columns
df_rank['Average_rank'] = df_rank[rank_cols].mean(axis=1)
df_rank=df_rank.sort_values('Average_rank')
df_rank['Final Rank']=range(1, len(df_rank)+1)


#Cleaning the data
df_rank.reset_index(inplace=True)
df_rank.columns.values[0] = "Ticker"
df_rank = df_rank.round(2)

print(df_rank.head(10))

#V Top 10 Stocks to Buy
top10 = df_rank.nsmallest(10, 'Final Rank').copy()
print("Top 10 stocks to buy based on the ranking:")
print(top10[['Ticker', 'Average_rank', 'Final Rank']])

#Visualization - Bar Plot of Top 10 Scores
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.bar(top10['Ticker'], top10['Average_rank'], color="skyblue")
plt.title("Top 10 Undervalued Stocks (lower average rank = better)")
plt.ylabel("Average rank ↓")
plt.xlabel("Ticker")
plt.grid(axis='y', linestyle='--', alpha=0.7)

#Add values on top of bars
for i, v in enumerate(top10['Average_rank']):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)

plt.tight_layout()
plot_path = PLOTS_DIR / "top10_barplot.png"
plt.savefig(plot_path)
print(f"Plot saved at: {plot_path}")

#VI Exit Strategy

my_stocks = ["GOOG","APA","AAPL", "JPM", "NKE", "REGN","TROW", "UNH", "DIS", "MRK"] #all stocks that I currently have in my portfolio
df_rank["Owned"] = df_rank["Ticker"].isin(my_stocks)

df_rank["Sell/Keep"] = df_rank.apply(lambda row: "Sell" if row["Owned"] and row["Drawdown (%)"] < 5
                                     else ("Keep" if row["Owned"] and row["Drawdown (%)"] >= 5 else ""),
                                     axis=1)
print(df_rank.head(10))

#VII Extracting data to CSV
df_final = df_rank.copy()
print(df_final.head(10))
df_final.to_csv(filepath_3, index = True)
print(f"The report has been finished and saved in{filepath_3}")

