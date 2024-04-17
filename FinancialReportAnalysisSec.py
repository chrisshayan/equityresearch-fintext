import yfinance as yf
from matplotlib import pyplot as plt
from pandas.tseries.offsets import DateOffset
from sec_api import ExtractorApi
import requests
import json
import numpy as np
from openai import OpenAI
import os
from utils import get_earnings_transcript, Raptor
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

from reportlab.lib import colors
from reportlab.lib import pagesizes
from reportlab.platypus import SimpleDocTemplate, Frame, Paragraph, Image, PageTemplate, FrameBreak, Spacer, Table, TableStyle, NextPageTemplate, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT

LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="..."
LANGCHAIN_PROJECT="llmops-sample"

ticker_symbol = "NVDA"  # The ticker symbol of the company. US stock only.
sec_api_key = os.environ.get("SEC_API_KEY")  # Your SEC API key, get it from https://sec-api.io/ for free.

#llm = "gpt-4-turbo-preview"
#llm = "llama2"
llm = "openchat"

# embd = OpenAIEmbeddings()
# create the open-source embedding function
embd = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
model = ChatOpenAI(temperature=0, model=llm)
rag_helper = Raptor(model, embd)

if 'gpt' in llm:
    print("Using OpenAI GPT")
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
else:
    print("Using local LLM, make sure you have installed Ollama (https://ollama.com/download) and have it running")
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',  # required, but unused
    )


class ReportAnalysis:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.stock = yf.Ticker(ticker_symbol)
        self.info = self.stock.info
        self.project_dir = f"projects/{ticker_symbol}/"
        os.makedirs(self.project_dir, exist_ok=True)
        self.extractor = ExtractorApi(sec_api_key)
        self.report_address = self.get_sec_report_address()

        # self.system_prompt_v3 = """
        self.system_prompt = f"""
            Role: Expert Investor in {self.stock.info['industry']}
            Department: Finance
            Primary Responsibility: Generation of Customized Financial Analysis Reports

            Role Description: As an Expert Investor within the finance domain, your expertise is harnessed to develop 
            bespoke Financial Analysis Reports that cater to specific client requirements. This role demands a deep 
            dive into financial statements and market data to unearth insights regarding a company's financial 
            performance and stability. Engaging directly with clients to gather essential information and 
            continuously refining the report with their feedback ensures the final product precisely meets their 
            needs and expectations. Generate reports similar quality to Goldman Sachs.

            Key Objectives:

            Analytical Precision: Employ meticulous analytical prowess to interpret financial data, identifying 
            underlying trends and anomalies. Effective Communication: Simplify and effectively convey complex 
            financial narratives, making them accessible and actionable to non-specialist audiences. Client Focus: 
            Dynamically tailor reports in response to client feedback, ensuring the final analysis aligns with their 
            strategic objectives. Adherence to Excellence: Maintain the highest standards of quality and integrity in 
            report generation, following established benchmarks for analytical rigor. Performance Indicators: The 
            efficacy of the Financial Analysis Report is measured by its utility in providing clear, actionable 
            insights. This encompasses aiding corporate decision-making, pinpointing areas for operational 
            enhancement, and offering a lucid evaluation of the company's financial health. Success is ultimately 
            reflected in the report's contribution to informed investment decisions and strategic planning.
            
            Technology Integration:

            Utilize advanced FinTech tools for data analysis, including: Sentiment Analysis Platforms: Leverage 
            AI-powered platforms to analyze public sentiment towards company and its products, gauging potential market 
            reception for upcoming releases. Alternative Data Providers: Access and analyze alternative data sets 
            sourced from web traffic, app downloads, and social media engagement to gain deeper insights into 
            consumer behavior and market trends. Financial Modeling Software: Employ sophisticated 
            financial modeling software to conduct scenario analyses, stress tests, and valuation calculations for 
            company stock. By including these specific examples, you demonstrate how the Expert Investor stays ahead 
            of the curve by leveraging cutting-edge FinTech tools to enrich their analysis and provide more 
            comprehensive insights for clients.

            Benchmarking:

            Utilize a multi-faceted approach to ensure the highest standards of analytical rigor: Industry Best 
            Practices: Apply financial valuation methodologies recommended by reputable institutions like Aswath 
            Damodaran or industry-specific valuation metrics relevant to the Consumer Electronics sector. Peer Group 
            Comparison: Benchmark Company's financial performance against its major competitors in sector, 
            identifying areas of strength and weakness. Regulatory Standards: Ensure all financial analysis adheres 
            to Generally Accepted Accounting Principles (GAAP) and discloses any potential conflicts of interest or 
            limitations in the analysis.
            
            """

    def get_stock_performance(self):
        def fetch_stock_data(ticker, period="1y"):
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist['Close']

        target_close = fetch_stock_data(self.ticker_symbol)
        sp500_close = fetch_stock_data("^GSPC")

        ticker_change = (target_close - target_close.iloc[0]) / target_close.iloc[0] * 100
        sp500_change = (sp500_close - sp500_close.iloc[0]) / sp500_close.iloc[0] * 100

        start_date = ticker_change.index.min()
        four_months = start_date + DateOffset(months=4)
        eight_months = start_date + DateOffset(months=8)
        end_date = ticker_change.index.max()

        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(14, 7))
        plt.plot(ticker_change.index, ticker_change, label=ticker_symbol + ' Change %', color='blue')
        plt.plot(sp500_change.index, sp500_change, label='S&P 500 Change %', color='red')

        plt.title(ticker_symbol + ' vs S&P 500 - Change % Over the Past Year')
        plt.xlabel('Date')
        plt.ylabel('Change %')

        plt.xticks([start_date, four_months, eight_months, end_date],
                   [start_date.strftime('%Y-%m'),
                    four_months.strftime('%Y-%m'),
                    eight_months.strftime('%Y-%m'),
                    end_date.strftime('%Y-%m')])

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plot_path = f"{self.project_dir}/stock_performance.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def get_sec_report_address(self):
        address_json = f"{self.project_dir}/sec_report_address.json"
        if not os.path.exists(address_json):
            endpoint = f"https://api.sec-api.io?token={sec_api_key}"

            # The query to find 10-K filings for a specific company
            query = {
                "query": {"query_string": {"query": f"ticker:{self.ticker_symbol} AND formType:\"10-K\""}},
                "from": "0",
                "size": "1",
                "sort": [{"filedAt": {"order": "desc"}}]
            }

            # Making the request to the SEC API
            response = requests.post(endpoint, json=query)

            if response.status_code == 200:
                # Parsing the response
                filings = response.json()['filings']
                if filings:
                    # Assuming the latest 10-K filing is what we want
                    latest_10k_url = filings[0]
                    print(f"Latest 10-K report URL for {self.ticker_symbol}: {latest_10k_url}")
                else:
                    print(f"No 10-K filings found for {self.ticker_symbol}.")
            else:
                print("Failed to retrieve filings from SEC API.")

            with open(address_json, "w") as f:
                json.dump(latest_10k_url, f)
        else:
            with open(address_json, "r") as f:
                latest_10k_url = json.load(f)

        return latest_10k_url['linkToFilingDetails']

    def get_key_data(self):
        # Fetch historical market data for the past 6 months
        hist = self.stock.history(period="6mo")

        info = self.info
        close_price = hist['Close'].iloc[-1]

        # Calculate the average daily trading volume
        avg_daily_volume_6m = hist['Volume'].mean()

        # Print the result
        # print(f"Over the past 6 months, the average daily trading volume for {ticker_symbol} was: {avg_daily_volume_6m:.2f}")
        result = {
            f"6m avg daily val ({info['currency']}mn)": "{:.2f}".format(avg_daily_volume_6m / 1e6),
            f"Closing Price ({info['currency']})": "{:.2f}".format(close_price),
            f"Market Cap ({info['currency']}mn)": "{:.2f}".format(info['marketCap'] / 1e6),
            f"52 Week Price Range ({info['currency']})": f"{info['fiftyTwoWeekLow']} - {info['fiftyTwoWeekHigh']}",
            f"BVPS ({info['currency']})": info['bookValue'],
            f"Stock Price Volatility Beta": info['beta'],
            f"Target Low Price ({info['currency']})": info['targetLowPrice'],
            f"Target Mean Price ({info['currency']})": info['targetMeanPrice'],
            f"Target Median Price ({info['currency']})": info['targetMedianPrice'],
            f"Target High Price ({info['currency']})": info['targetHighPrice'],
        }
        return result

    def get_company_info(self):
        info = self.info
        result = {
            "Company Name": info['shortName'],
            "Industry": info['industry'],
            "Sector": info['sector'],
            "Country": info['country'],
            "Website": info['website']
        }
        return result

    def get_income_stmt(self):
        income_stmt = self.stock.financials
        return income_stmt

    def get_balance_sheet(self):
        balance_sheet = self.stock.balance_sheet
        return balance_sheet

    def get_cash_flow(self):
        cash_flow = self.stock.cashflow
        return cash_flow

    def get_analyst_recommendations(self):
        recommendations = self.stock.recommendations
        row_0 = recommendations.iloc[0, 1:]  # Exclude 'period' column

        # Find the maximum voting result
        max_votes = row_0.max()
        majority_voting_result = row_0[row_0 == max_votes].index.tolist()

        return majority_voting_result[0], max_votes

    def get_earnings(self, quarter, year):
        earnings = get_earnings_transcript(quarter, self.ticker_symbol, year)
        return earnings

    def get_10k_section(self, section):
        """
            Get 10-K reports from SEC EDGAR
        """
        if section not in [1, "1A", "1B", 2, 3, 4, 5, 6, 7, "7A", 8, 9, "9A", "9B", 10, 11, 12, 13, 14, 15]:
            raise ValueError(
                "Section must be in [1, 1A, 1B, 2, 3, 4, 5, 6, 7, 7A, 8, 9, 9A, 9B, 10, 11, 12, 13, 14, 15]")

        section = str(section)
        os.makedirs(f"{self.project_dir}/10k", exist_ok=True)

        report_name = f"{self.project_dir}/10k/section_{section}.txt"

        if not os.path.exists(report_name):
            section_text = self.extractor.get_section(self.report_address, section, "text")

            with open(report_name, "w") as f:
                f.write(section_text)
        else:
            with open(report_name, "r") as f:
                section_text = f.read()

        return section_text

    def get_10k_rag(self, section):
        # Now, use all_texts to build the vectorstore with Chroma
        vector_dir = f"{self.project_dir}/10k/section_{section}_vectorstore"
        if not os.path.exists(vector_dir):
            section_text = self.get_10k_section(section)
            all_texts = rag_helper.text_spliter(section_text, chunk_size_tok=2000, level=1, n_levels=3)

            vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd, persist_directory=vector_dir)
            vectorstore.persist()
        else:
            vectorstore = Chroma(persist_directory=vector_dir, embedding_function=embd)
            vectorstore.get()
        retriever = vectorstore.as_retriever()

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # Chain
        rag_chain = (
            # {"context": retriever | format_docs, "question": RunnablePassthrough()}
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
        )

        # Question
        # rag_chain.invoke("What is the profit of the company. you should not say you don't know because all the required information is in the context")
        # rag_chain.invoke("Analyse the income statement of the company for the year 2023")
        return rag_chain

    def analyze_income_stmt(self):
        income_stmt = self.get_income_stmt()
        df_string = "Income statement:" + income_stmt.to_string().strip()

        question = ("Embark on a thorough analysis of the company's income statement for the current fiscal year, "
                    "focusing on revenue streams, cost of goods sold, operating expenses, and net income to discern "
                    "the operational performance and profitability. Examine the gross profit margin to understand the "
                    "cost efficiency, operating margin for operational effectiveness, and net profit margin to assess "
                    "overall profitability. Compare these financial metrics against historical data to identify "
                    "growth patterns, profitability trends, and operational challenges. Conclude with a strategic "
                    "overview of the company's financial health, offering insights into revenue growth sustainability "
                    "and potential areas for cost optimization and profit maximization in a single paragraph. Less "
                    "than 130 words.")

        question_v4 = ("Analyze the company's current year income statement, focusing on revenue streams, costs, "
                    "and profitability metrics (gross margin, operating margin, net margin). Compare these metrics to "
                    "historical data to identify trends and challenges.  In a concise summary (less than 1 sentence), "
                    "assess the company's financial health, including revenue growth sustainability and opportunities "
                    "for cost optimization and profit improvement.")

        answer = self.ask_question(question, 7, df_string, use_rag=False)
        return answer

    def analyze_balance_sheet(self):
        balance_sheet = self.get_balance_sheet()
        df_string = "Balance sheet:" + balance_sheet.to_string().strip()

        question = ("Delve into a detailed scrutiny of the company's balance sheet for the most recent fiscal year, "
                    "pinpointing the structure of assets, liabilities, and shareholders' equity to decode the firm's "
                    "financial stability and operational efficiency. Focus on evaluating the liquidity through "
                    "current assets versus current liabilities, the solvency via long-term debt ratios, "
                    "and the equity position to gauge long-term investment potential. Contrast these metrics with "
                    "previous years' data to highlight financial trends, improvements, or deteriorations. Finalize "
                    "with a strategic assessment of the company's financial leverage, asset management, and capital "
                    "structure, providing insights into its fiscal health and future prospects in a single paragraph. "
                    "Less than 130 words.")

        question_v4 = ("Analyze the company's latest balance sheet, dissecting assets, liabilities, and equity to assess "
                    "financial stability and efficiency. Evaluate liquidity (current ratio), solvency (debt ratios), "
                    "and shareholder equity for long-term viability. Compare these metrics to historical data to "
                    "identify trends. In a concise summary (less than 1 sentence), assess the company's financial "
                    "leverage, asset management, and capital structure, offering insights into its financial health "
                    "and future prospects.")

        answer = self.ask_question(question, 7, df_string, use_rag=False)
        return answer

    def analyze_cash_flow(self):
        cash_flow = self.get_cash_flow()
        df_string = "Balance sheet:" + cash_flow.to_string().strip()

        question = ("Dive into a comprehensive evaluation of the company's cash flow for the latest fiscal year, "
                    "focusing on cash inflows and outflows across operating, investing, and financing activities. "
                    "Examine the operational cash flow to assess the core business profitability, scrutinize "
                    "investing activities for insights into capital expenditures and investments, and review "
                    "financing activities to understand debt, equity movements, and dividend policies. Compare these "
                    "cash movements to prior periods to discern trends, sustainability, and liquidity risks. Conclude "
                    "with an informed analysis of the company's cash management effectiveness, liquidity position, "
                    "and potential for future growth or financial challenges in a single paragraph. Less than 130 "
                    "words.")

        question_v4 = ("Analyze the company's latest cash flow statement, focusing on operating, investing, "
                    "and financing activities.  Evaluate operating cash flow for business profitability, "
                    "investing activities for capital allocation, and financing activities for debt, equity, "
                    "and dividends. Compare these to historical data to identify trends and risks. In a concise "
                    "summary (less than 1 sentence), assess the company's cash flow management, liquidity, "
                    "and potential for future growth or challenges.")

        answer = self.ask_question(question, 7, df_string, use_rag=False)
        return answer

    def financial_summarization(self):
        income_stmt_analysis = self.analyze_income_stmt()
        balance_sheet_analysis = self.analyze_balance_sheet()
        cash_flow_analysis = self.analyze_cash_flow()

        question = (f"Income statement analysis: {income_stmt_analysis}, \
        Balance sheet analysis: {balance_sheet_analysis}, \
        Cash flow analysis: {cash_flow_analysis}, \ Synthesize the findings from the in-depth analysis of the income "
                    f"statement, balance sheet, and cash flow for the latest fiscal year. Highlight the core insights "
                    f"regarding the company's operational performance, financial stability, and cash management "
                    f"efficiency. Discuss the interrelations between revenue growth, cost management strategies, "
                    f"and their impact on profitability as revealed by the income statement. Incorporate the balance "
                    f"sheet's insights on financial structure, liquidity, and solvency to provide a comprehensive "
                    f"view of the company's financial health. Merge these with the cash flow analysis to illustrate "
                    f"the company's liquidity position, investment activities, and financing strategies. Conclude "
                    f"with a holistic assessment of the company's fiscal health, identifying strengths, "
                    f"potential risks, and strategic opportunities for growth and stability. Offer recommendations to "
                    f"address identified challenges and capitalize on the opportunities to enhance shareholder value "
                    f"in a single paragraph. Less than 150 words.")

        question_woan = (f"Income statement analysis: {income_stmt_analysis}, \
        Balance sheet analysis: {balance_sheet_analysis}, \
        Cash flow analysis: {cash_flow_analysis}, \ Synthesize the findings from the in-depth analysis of the income "
                    f"statement, balance sheet, and cash flow for the latest fiscal year. "
                    f". Discuss company's competitive strategy, to what extend does it align or "
                       f"differ from its competitors in same segment. Using your assessment of industry KIRs ("
                       f"Key Industry risks) identify operating cycle (supply, production, demand, and collection)"
                       f"and specify how they relate to the company. "
                       f"What are the company competencies and weakness?"
                       f"How effectively company manage its net working assets? Consider DOH ratios, sales/NWA,"
                       f"changes in NWCC, and compared to peers in same industry and how the ratios you calculated. "
                       f"have NCAO and CADA been sufficient to cover maintenance capex? what level of CAPEX do"
                       f"you anticipate going forward?"
                    f"You have no limit on how many words you want to generate.")


        #woan_answer = self.ask_question(question_woan, 7, use_rag=False)
        #print("Woan's Answer:")
        #print(woan_answer)

        answer = self.ask_question(question, 7, use_rag=False)
        return {"Income Statement Analysis": income_stmt_analysis, "Balance Sheet Analysis": balance_sheet_analysis,
                "Cash Flow Analysis": cash_flow_analysis, "Financial Summary": answer}

    def ask_question(self, question, section, table_str=None, use_rag=False):
        if use_rag:
            rag_chain = self.get_10k_rag(section)
            if table_str:
                prompt = f"{self.system_prompt}\n\n{table_str}\n\nQuestion: {question}"
            else:
                prompt = f"{self.system_prompt}\n\nQuestion: {question}"
            answer = rag_chain.invoke(prompt)
        else:
            section_text = self.get_10k_section(7)
            if table_str:
                prompt = f"{self.system_prompt}\n\n{table_str}\n\nResource: {section_text}\n\nQuestion: {question}"
            else:
                prompt = f"{self.system_prompt}\n\nResource: {section_text}\n\nQuestion: {question}"

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    }
                ],
                model=llm,
                temperature=0,
                max_tokens=300,
                # response_format={ "type": "json_object" },
            )
            answer = chat_completion.choices[0].message.content

        return answer


ra = ReportAnalysis(ticker_symbol)
answer = ra.financial_summarization()
answer.keys()

page_width, page_height = pagesizes.A4
left_column_width = page_width * 2/3
right_column_width = page_width - left_column_width
margin = 4

pdf_path = os.path.join(ra.project_dir, f"{ticker_symbol}_report_{llm}-v4.pdf")
doc = SimpleDocTemplate(pdf_path, pagesize=pagesizes.A4)

frame_left = Frame(margin, margin, left_column_width-margin*2, page_height-margin*2, id='left')
frame_right = Frame(left_column_width, margin, right_column_width-margin*2, page_height-margin*2, id='right')

# single_frame = Frame(margin, margin, page_width-margin*2, page_height-margin*2, id='single')
# single_column_layout = PageTemplate(id='OneCol', frames=[single_frame])

left_column_width_p2 = (page_width-margin*3) // 2
right_column_width_p2 = left_column_width_p2
frame_left_p2 = Frame(margin, margin, left_column_width_p2-margin*2, page_height-margin*2, id='left')
frame_right_p2 = Frame(left_column_width_p2, margin, right_column_width_p2-margin*2, page_height-margin*2, id='right')

page_template = PageTemplate(id='TwoColumns', frames=[frame_left, frame_right])
page_template_p2 = PageTemplate(id='TwoColumns_p2', frames=[frame_left_p2, frame_right_p2])
doc.addPageTemplates([page_template, page_template_p2])

styles = getSampleStyleSheet()

custom_style = ParagraphStyle(
    name="Custom",
    parent=styles['Normal'],
    fontName="Helvetica",
    fontSize=10,
    # leading=15,
    alignment=TA_JUSTIFY,
)

title_style = ParagraphStyle(
    name="TitleCustom",
    parent=styles['Title'],
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=20,
    alignment=TA_LEFT,
    spaceAfter=10,
)

subtitle_style = ParagraphStyle(
    name="Subtitle",
    parent=styles['Heading2'],
    fontName="Helvetica-Bold",
    fontSize=14,
    leading=12,
    alignment=TA_LEFT,
    spaceAfter=6,
)

content = []
content.append(Paragraph(f"Equity Research Report Using FinGPT: {ra.get_company_info()['Company Name']}", title_style))

content.append(Paragraph("Income Statement Analysis", subtitle_style))
content.append(Paragraph(answer['Income Statement Analysis'], custom_style))

content.append(Paragraph("Balance Sheet Analysis", subtitle_style))
content.append(Paragraph(answer['Balance Sheet Analysis'], custom_style))

content.append(Paragraph("Cashflow Analysis", subtitle_style))
content.append(Paragraph(answer['Cash Flow Analysis'], custom_style))

content.append(Paragraph("Summarization", subtitle_style))
content.append(Paragraph(answer['Financial Summary'], custom_style))


content.append(FrameBreak())

table_style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.white),
    ('BACKGROUND', (0, 0), (-1, 0), colors.white),
    ('FONT', (0, 0), (-1, -1), 'Helvetica', 8),
    ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 12),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ALIGN', (0,1), (0,-1), 'LEFT'),
    ('ALIGN', (1,1), (1,-1), 'RIGHT'),
    ('LINEBELOW', (0,0), (-1,0), 2, colors.black),
])
full_length = right_column_width-2*margin

rating, _ = ra.get_analyst_recommendations()

data = [["Rating:", rating.upper()]]
col_widths = [full_length//3*2, full_length//3]
table = Table(data, colWidths=col_widths)
table.setStyle(table_style)
content.append(table)

# content.append(Paragraph("", custom_style))
content.append(Spacer(1, 0.15*inch))
key_data = ra.get_key_data()
data = [["Key data", ""]]
data += [
    [k, v] for k, v in key_data.items()
]
col_widths = [full_length//3*2, full_length//3]
table = Table(data, colWidths=col_widths)
table.setStyle(table_style)
content.append(table)


# Matplotlib

data = [["Share Performance"]]
col_widths = [full_length]
table = Table(data, colWidths=col_widths)
table.setStyle(table_style)
content.append(table)

plot_path = ra.get_stock_performance()
width = right_column_width
height = width//2
content.append(Image(plot_path, width=width, height=height))


content.append(NextPageTemplate('TwoColumns_p2'))
content.append(PageBreak())

table_style2 = TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.white),
    ('BACKGROUND', (0, 0), (-1, 0), colors.white),
    ('FONT', (0, 0), (-1, -1), 'Helvetica', 6),
    ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ALIGN', (0,1), (0,-1), 'LEFT'),
    ('ALIGN', (1,1), (1,-1), 'RIGHT'),
    ('LINEBELOW', (0,0), (-1,0), 2, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 2, colors.black),
])


df = ra.get_income_stmt()
df = df[df.columns[:3]]
def convert_if_money(value):
    if np.abs(value) >= 1000000:
        return value / 1000000
    else:
        return value

df = df.applymap(convert_if_money)

df.columns = [col.strftime('%Y') for col in df.columns]
df.reset_index(inplace=True)
currency = ra.info['currency']
df.rename(columns={'index': f'FY ({currency} mn)'}, inplace=True)
table_data = [["Income Statement"]]
table_data += [df.columns.to_list()] + df.values.tolist()

table = Table(table_data)
table.setStyle(table_style2)
content.append(table)

content.append(FrameBreak())

df = ra.get_cash_flow()
df = df[df.columns[:3]]

df = df.applymap(convert_if_money)

df.columns = [col.strftime('%Y') for col in df.columns]
df.reset_index(inplace=True)
currency = ra.info['currency']
df.rename(columns={'index': f'FY ({currency} mn)'}, inplace=True)
table_data = [["Cash Flow Sheet"]]
table_data += [df.columns.to_list()] + df.values.tolist()

table = Table(table_data)
table.setStyle(table_style2)
content.append(table)
# content.append(Paragraph('This is a single column on the second page', custom_style))
# content.append(Spacer(1, 0.2*inch))
# content.append(Paragraph('More content in the single column.', custom_style))

# build the pdf
doc.build(content)
