import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- INSTRUCTIONS FOR CREATING AN EXECUTABLE ---
# 1. Make sure you have Python installed on your system.
# 2. Install the required libraries by opening a terminal or command prompt and running:
#    pip install matplotlib pyinstaller
# 3. Save this script as a Python file (e.g., "finance_app.py").
# 4. In the terminal, navigate to the directory where you saved the file.
# 5. Run the following command to create a single executable file:
#    pyinstaller --onefile --windowed finance_app.py
# 6. Look in the "dist" folder that pyinstaller creates. Your executable file will be inside.

class FinanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Personal Finance & Savings Goal Tracker")
        self.geometry("1100x800")
        self.minsize(950, 750)

        # --- Style ---
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook.Tab", font=('Helvetica', 11, 'bold'))
        style.configure("TLabel", font=('Helvetica', 10))
        style.configure("TButton", font=('Helvetica', 10, 'bold'))
        style.configure("Header.TLabel", font=('Helvetica', 14, 'bold'), padding=(0, 10, 0, 10))
        style.configure("Result.TLabel", font=('Helvetica', 11, 'bold'), foreground="#006400")
        style.configure("Negative.Result.TLabel", font=('Helvetica', 11, 'bold'), foreground="#C00000")
        
        # --- Data ---
        self.create_data_dictionaries()

        # --- Create Tabs ---
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.calc_frame = ttk.Frame(notebook, padding="10")
        self.budget_frame = ttk.Frame(notebook, padding="10")
        self.student_frame = ttk.Frame(notebook, padding="10")
        self.savings_frame = ttk.Frame(notebook, padding="10")

        notebook.add(self.calc_frame, text='Income Calculator')
        notebook.add(self.budget_frame, text='Monthly Budget')
        notebook.add(self.savings_frame, text='Savings Goal')
        notebook.add(self.student_frame, text='Student Finance Tracker')

        self.create_calculator_tab()
        self.create_budget_tab()
        self.create_savings_goal_tab()
        self.create_student_finance_tab()
        
        self.calculate_all() # Initial calculation on startup

    def create_data_dictionaries(self):
        """Initialize all tkinter variables and data structures."""
        self.calc_vars = {
            'salary': tk.DoubleVar(value=50000), 'performance_bonus': tk.DoubleVar(value=0),
            'sign_on_bonus': tk.DoubleVar(value=0), 'pension_contrib': tk.DoubleVar(value=5.0),
            'salary_sacrifice': tk.DoubleVar(value=0), 'student_loan_plan': tk.StringVar(value='Plan 2'),
            'voluntary_sl': tk.DoubleVar(value=0)
        }
        self.budget_vars = {
            'Rent / Mortgage': tk.DoubleVar(value=1200), 'Council Tax': tk.DoubleVar(value=150),
            'Utilities': tk.DoubleVar(value=200), 'Groceries': tk.DoubleVar(value=400),
            'Transport': tk.DoubleVar(value=100), 'Subscriptions': tk.DoubleVar(value=30),
            'Phone & Internet': tk.DoubleVar(value=50), 'Tithing': tk.DoubleVar(value=0),
            'Savings / Investments': tk.DoubleVar(value=250), 'Other Debt Repayments': tk.DoubleVar(value=0),
            'Other / Discretionary': tk.DoubleVar(value=300)
        }
        self.student_vars = {
            'initial_balance': tk.DoubleVar(value=45000), 'course_start_year': tk.IntVar(value=2020),
            'salary_growth': tk.DoubleVar(value=3.0)
        }
        self.savings_vars = {
            'goal_name': tk.StringVar(value="New Car"), 'goal_amount': tk.DoubleVar(value=15000),
            'current_savings': tk.DoubleVar(value=2000)
        }
        self.chart_options = {
            'chart_type': tk.StringVar(value='Pie Chart'),
            'explode_category': tk.StringVar(value='None'),
            'show_labels': tk.BooleanVar(value=True)
        }
        self.rates = {
            'Plan 1': {'threshold': 24990, 'rate': 0.09, 'interest': 0.043, 'write_off': 25},
            'Plan 2': {'threshold': 27295, 'rate': 0.09, 'interest': 0.073, 'write_off': 30},
            'Plan 4': {'threshold': 31395, 'rate': 0.09, 'interest': 0.043, 'write_off': 30},
            'Postgraduate': {'threshold': 21000, 'rate': 0.06, 'interest': 0.073, 'write_off': 30},
            'None': {'threshold': 9999999, 'rate': 0, 'interest': 0, 'write_off': 0}
        }
        self.calc_results = {}

    def create_calculator_tab(self):
        frame = self.calc_frame
        frame.grid_columnconfigure(1, weight=1)
        ttk.Label(frame, text="Income & Deductions Calculator", style="Header.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        
        inputs_frame = ttk.LabelFrame(frame, text="Inputs", padding=10)
        inputs_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        labels = ["Annual Salary (£)", "Performance Bonus (£)", "Sign-on Bonus (£)", "Pension Contribution (%)", "Salary Sacrifice (£)", "Student Loan Plan", "Voluntary SL Repayment (£/month)"]
        vars_keys = ['salary', 'performance_bonus', 'sign_on_bonus', 'pension_contrib', 'salary_sacrifice', 'student_loan_plan', 'voluntary_sl']
        for i, label_text in enumerate(labels):
            ttk.Label(inputs_frame, text=label_text).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            if label_text == "Student Loan Plan":
                ttk.Combobox(inputs_frame, textvariable=self.calc_vars['student_loan_plan'], values=list(self.rates.keys())).grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            else:
                ttk.Entry(inputs_frame, textvariable=self.calc_vars[vars_keys[i]]).grid(row=i, column=1, sticky="ew", padx=5, pady=2)
        
        results_frame = ttk.LabelFrame(frame, text="Results (Annual)", padding=10)
        results_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.result_labels = {}
        result_keys = ["Gross Annual Income", "Adjusted Gross Income", "Taxable Income", "---Deductions---", "Income Tax", "National Insurance", "Student Loan (Mandatory)", "---Summary---", "Net Annual Income", "Net Monthly Income"]
        for i, key in enumerate(result_keys):
            ttk.Label(results_frame, text=f"{key}:").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            if "---" not in key:
                self.result_labels[key] = ttk.Label(results_frame, text="£0.00", style="Result.TLabel")
                self.result_labels[key].grid(row=i, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Button(frame, text="Calculate All", command=self.calculate_all).grid(row=2, column=0, columnspan=2, pady=10)

    def calculate_all(self):
        try:
            salary = self.calc_vars['salary'].get()
            perf_bonus = self.calc_vars['performance_bonus'].get()
            sign_on_bonus = self.calc_vars['sign_on_bonus'].get()
            pension_percent = self.calc_vars['pension_contrib'].get() / 100
            sacrifice = self.calc_vars['salary_sacrifice'].get()
            plan = self.calc_vars['student_loan_plan'].get()
            
            gross_income = salary + perf_bonus + sign_on_bonus
            adjusted_gross = gross_income - (gross_income * pension_percent) - sacrifice
            personal_allowance = 12570
            if adjusted_gross > 100000: personal_allowance = max(0, 12570 - ((adjusted_gross - 100000) / 2))
            taxable_income = max(0, adjusted_gross - personal_allowance)
            tax = 0
            if taxable_income > 125140: tax += (taxable_income - 125140) * 0.45
            if taxable_income > 37700: tax += (min(taxable_income, 125140) - 37700) * 0.40
            if taxable_income > 0: tax += min(taxable_income, 37700) * 0.20
            ni = 0
            if adjusted_gross > 50270: ni += (adjusted_gross - 50270) * 0.02
            if adjusted_gross > 12570: ni += (min(adjusted_gross, 50270) - 12570) * 0.08
            plan_details = self.rates[plan]
            mandatory_sl = (adjusted_gross - plan_details['threshold']) * plan_details['rate'] if adjusted_gross > plan_details['threshold'] else 0
            total_deductions = tax + ni + mandatory_sl
            net_annual = adjusted_gross - total_deductions
            net_monthly = net_annual / 12

            self.calc_results.update({'net_monthly': net_monthly, 'mandatory_sl_annual': mandatory_sl, 'voluntary_sl_annual': self.calc_vars['voluntary_sl'].get() * 12})
            self.result_labels["Gross Annual Income"].config(text=f"£{gross_income:,.2f}")
            self.result_labels["Adjusted Gross Income"].config(text=f"£{adjusted_gross:,.2f}")
            self.result_labels["Taxable Income"].config(text=f"£{taxable_income:,.2f}")
            self.result_labels["Income Tax"].config(text=f"£{tax:,.2f}")
            self.result_labels["National Insurance"].config(text=f"£{ni:,.2f}")
            self.result_labels["Student Loan (Mandatory)"].config(text=f"£{mandatory_sl:,.2f}")
            self.result_labels["Net Annual Income"].config(text=f"£{net_annual:,.2f}")
            self.result_labels["Net Monthly Income"].config(text=f"£{net_monthly:,.2f}")
            
            self.update_budget()
            self.update_savings_goal()
        except tk.TclError:
            messagebox.showerror("Invalid Input", "Please ensure all input fields contain valid numbers.")

    def create_budget_tab(self):
        frame = self.budget_frame
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        ttk.Label(frame, text="Monthly Budget Planner", style="Header.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        
        left_frame = ttk.Frame(frame)
        left_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ns")

        inputs_frame = ttk.LabelFrame(left_frame, text="Monthly Outgoings", padding=10)
        inputs_frame.pack(fill="x")
        for i, (name, var) in enumerate(self.budget_vars.items()):
            ttk.Label(inputs_frame, text=f"{name} (£):").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(inputs_frame, textvariable=var, width=15).grid(row=i, column=1, sticky="ew", padx=5, pady=2)
        
        options_frame = ttk.LabelFrame(left_frame, text="Chart Options", padding=10)
        options_frame.pack(fill="x", pady=10)
        ttk.Label(options_frame, text="Chart Type:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(options_frame, textvariable=self.chart_options['chart_type'], values=['Pie Chart', 'Bar Chart']).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(options_frame, text="Explode Slice:").grid(row=1, column=0, sticky="w", pady=2)
        self.explode_combo = ttk.Combobox(options_frame, textvariable=self.chart_options['explode_category'], values=['None'] + list(self.budget_vars.keys()))
        self.explode_combo.grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Checkbutton(options_frame, text="Show Labels", variable=self.chart_options['show_labels']).grid(row=2, column=0, columnspan=2, sticky="w")

        ttk.Button(left_frame, text="Update Budget & Chart", command=self.update_budget).pack(pady=10)

        summary_frame = ttk.LabelFrame(frame, text="Summary & Chart", padding=10)
        summary_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        summary_frame.grid_rowconfigure(4, weight=1)
        summary_frame.grid_columnconfigure(0, weight=1)
        
        self.budget_result_labels = {}
        summary_keys = ["Monthly Take-Home Pay", "Total Monthly Outgoings", "Disposable Income"]
        for i, key in enumerate(summary_keys):
            ttk.Label(summary_frame, text=f"{key}:").grid(row=i, column=0, sticky="w", padx=5, pady=5)
            self.budget_result_labels[key] = ttk.Label(summary_frame, text="£0.00", style="Result.TLabel")
            self.budget_result_labels[key].grid(row=i, column=1, sticky="w", padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(); self.canvas = FigureCanvasTkAgg(self.fig, master=summary_frame)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=2, pady=10, sticky="nsew")
        
    def update_budget(self):
        net_monthly = self.calc_results.get('net_monthly', 0)
        total_outgoings = sum(var.get() for var in self.budget_vars.values())
        disposable = net_monthly - total_outgoings
        self.budget_result_labels["Monthly Take-Home Pay"].config(text=f"£{net_monthly:,.2f}")
        self.budget_result_labels["Total Monthly Outgoings"].config(text=f"£{total_outgoings:,.2f}")
        self.budget_result_labels["Disposable Income"].config(text=f"£{disposable:,.2f}", style="Result.TLabel" if disposable >= 0 else "Negative.Result.TLabel")
        
        self.ax.clear()
        labels = [name for name, var in self.budget_vars.items() if var.get() > 0]
        sizes = [var.get() for var in self.budget_vars.values() if var.get() > 0]
        
        if not sizes:
            self.canvas.draw()
            return
            
        chart_type = self.chart_options['chart_type'].get()
        if chart_type == 'Pie Chart':
            explode_cat = self.chart_options['explode_category'].get()
            explode = [0.1 if cat == explode_cat else 0 for cat in labels]
            autopct = '%1.1f%%' if self.chart_options['show_labels'].get() else None
            self.ax.pie(sizes, labels=labels, autopct=autopct, startangle=90, explode=explode)
            self.ax.axis('equal'); self.ax.set_title("Outgoings Breakdown")
        elif chart_type == 'Bar Chart':
            self.ax.barh(labels, sizes)
            self.ax.set_xlabel('Amount (£)')
            self.ax.set_title("Outgoings Breakdown")
            self.fig.tight_layout()

        self.canvas.draw()
        
    def create_savings_goal_tab(self):
        frame = self.savings_frame
        ttk.Label(frame, text="Savings Goal Tracker", style="Header.TLabel").pack(anchor="w")
        inputs_frame = ttk.LabelFrame(frame, text="Your Goal", padding=10)
        inputs_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(inputs_frame, text="Goal Name:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(inputs_frame, textvariable=self.savings_vars['goal_name'], width=30).grid(row=0, column=1, sticky="ew")
        ttk.Label(inputs_frame, text="Goal Amount (£):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(inputs_frame, textvariable=self.savings_vars['goal_amount']).grid(row=1, column=1, sticky="ew")
        ttk.Label(inputs_frame, text="Current Savings (£):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(inputs_frame, textvariable=self.savings_vars['current_savings']).grid(row=2, column=1, sticky="ew")
        ttk.Button(inputs_frame, text="Update Goal", command=self.update_savings_goal).grid(row=3, column=0, columnspan=2, pady=10)
        
        progress_frame = ttk.LabelFrame(frame, text="Progress", padding=10)
        progress_frame.pack(fill="x", padx=5, pady=5)
        self.savings_progress = ttk.Progressbar(progress_frame, orient="horizontal", length=400, mode="determinate")
        self.savings_progress.pack(pady=10, fill="x")
        self.savings_label = ttk.Label(progress_frame, text="Update to see progress", font=('Helvetica', 12, 'bold'))
        self.savings_label.pack(pady=5)

    def update_savings_goal(self):
        goal = self.savings_vars['goal_amount'].get()
        current = self.savings_vars['current_savings'].get()
        if goal > 0:
            percent_complete = (current / goal) * 100
            remaining = goal - current
            self.savings_progress['value'] = percent_complete
            self.savings_label.config(text=f"{percent_complete:.1f}% Complete! You have £{remaining:,.2f} left to save for your {self.savings_vars['goal_name'].get()}.")
        else:
            self.savings_progress['value'] = 0
            self.savings_label.config(text="Please set a goal amount greater than zero.")

    def create_student_finance_tab(self):
        frame = self.student_frame
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        top_frame = ttk.Frame(frame)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(top_frame, text="Student Finance Projections", style="Header.TLabel").pack(side="left", anchor="w")
        
        inputs_frame = ttk.LabelFrame(top_frame, text="Projection Inputs", padding=10)
        inputs_frame.pack(side="left", fill="x", expand=True, padx=20)
        ttk.Label(inputs_frame, text="Initial Loan Balance (£):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(inputs_frame, textvariable=self.student_vars['initial_balance']).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(inputs_frame, text="Course Start Year:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(inputs_frame, textvariable=self.student_vars['course_start_year']).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(inputs_frame, text="Annual Salary Growth (%):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(inputs_frame, textvariable=self.student_vars['salary_growth']).grid(row=2, column=1, sticky="ew", pady=2)
        ttk.Button(inputs_frame, text="Project Loan Repayment", command=self.project_loan).grid(row=3, column=0, columnspan=2, pady=10)
        
        summary_frame = ttk.LabelFrame(top_frame, text="Summary", padding=10)
        summary_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.student_summary_label = ttk.Label(summary_frame, text="Results will appear here.", wraplength=250)
        self.student_summary_label.pack()

        tree_frame = ttk.Frame(frame); tree_frame.grid(row=1, column=0, sticky="nsew", pady=10, padx=5)
        tree_frame.grid_rowconfigure(0, weight=1); tree_frame.grid_columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(tree_frame, columns=('Year', 'Age', 'Opening', 'Interest', 'Mandatory', 'Voluntary', 'Closing'), show='headings')
        for col in self.tree['columns']: self.tree.heading(col, text=col); self.tree.column(col, width=100, anchor='e')
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview); self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew"); vsb.grid(row=0, column=1, sticky="ns")
        
        chart_frame = ttk.Frame(frame); chart_frame.grid(row=1, column=1, sticky="nsew", pady=10, padx=5)
        chart_frame.grid_rowconfigure(0, weight=1); chart_frame.grid_columnconfigure(0, weight=1)
        self.fig_student, self.ax_student = plt.subplots(); self.canvas_student = FigureCanvasTkAgg(self.fig_student, master=chart_frame)
        self.canvas_student.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def project_loan(self):
        self.tree.delete(*self.tree.get_children())
        plan_name = self.calc_vars['student_loan_plan'].get()
        if plan_name == 'None': self.student_summary_label.config(text="No student loan plan selected."); return
        plan = self.rates[plan_name]
        balance = self.student_vars['initial_balance'].get()
        start_year = self.student_vars['course_start_year'].get()
        growth_rate = self.student_vars['salary_growth'].get() / 100
        current_salary = self.calc_vars['salary'].get()
        voluntary_repayment = self.calc_vars['voluntary_sl'].get() * 12
        total_interest_paid = 0; cleared_year = "Written Off"; loan_history = []
        for year_num in range(1, 41):
            loan_history.append(balance)
            if balance <= 0: cleared_year = start_year + year_num - 2; break
            if year_num > plan['write_off']: balance = 0; break
            projected_salary = current_salary * ((1 + growth_rate) ** (year_num - 1))
            interest_accrued = balance * plan['interest']; total_interest_paid += interest_accrued
            mandatory_repayment = (projected_salary - plan['threshold']) * plan['rate'] if projected_salary > plan['threshold'] else 0
            opening_balance = balance
            balance += interest_accrued - mandatory_repayment - voluntary_repayment
            balance = max(0, balance)
            self.tree.insert('', 'end', values=(f"{start_year + year_num -1}", f"{year_num}", f"£{opening_balance:,.2f}", f"£{interest_accrued:,.2f}", f"£{mandatory_repayment:,.2f}", f"£{voluntary_repayment:,.2f}", f"£{balance:,.2f}"))
        if balance <= 0 and cleared_year == "Written Off": cleared_year = start_year + year_num -1
        self.student_summary_label.config(text=f"Loan Cleared In Year: {cleared_year}\n\nTotal Interest Paid: £{total_interest_paid:,.2f}")
        self.ax_student.clear(); self.ax_student.plot(range(len(loan_history)), loan_history); self.ax_student.set_title("Loan Balance Over Time"); self.ax_student.set_xlabel("Years from Start"); self.ax_student.set_ylabel("Loan Balance (£)"); self.ax_student.grid(True); self.canvas_student.draw()

if __name__ == "__main__":
    try:
        app = FinanceApp()
        app.mainloop()
    except ImportError:
        messagebox.showerror("Dependency Error", "Matplotlib is required to run this application.\nPlease install it by running: pip install matplotlib")
