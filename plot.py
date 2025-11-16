import matplotlib.pyplot as plt

# Data
years = [2010, 2011, 2012, 2013, 2014]
sales = [100, 120, 110, 130, 150]

# Create a line plot
plt.plot(years, sales, marker='o', linestyle='-', color='blue')

# Add labels and title
plt.xlabel("Year")
plt.ylabel("Sales (Units)")
plt.title("Annual Sales Trend")

# Display the plot
plt.show()