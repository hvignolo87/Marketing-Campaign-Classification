# Marketing Campaing - Classification with Machine Learning

## Problem description
This work records real data from telemarketing campaigns to sell time deposits carried out by a Portuguese retail bank. Within a campaign, the operators offer the products and/or services in two ways and at different times: making phone calls (outgoing) or taking advantage of a customer's call center call for any other reason (incoming). Therefore, the result is a binary result contact: success/positive (buys the product) or failure/negative (does not buy it).

The data corresponds to contacts with clients made from May 2008 to June 2013 and total 35,010 records. Each record includes the result of the contact ("success", "failure") and different predictor variables or attributes. These include telemarketing attributes (eg. call address), product details (eg. interest rate offered) and customer information (eg. age). These records were enriched with data of a socio-economic nature (for example, the unemployment rate).

## Features description:

1. Bank customer data:
   
   - ```Age```: (numeric)
   - ```Job```: type of job (categorical: 'Worker', 'Manager', 'Administrative', 'Self-employed', 'Services', 'Technician', 'unemployed', 'Businessman', 'Retired', 'Housewife ', 'unknown', 'student')
   - ```Marital status```: marital status (categorical: 'married', 'divorced', 'single', 'unknown')
   - ```Education```: (categorical: 'basica.9y', 'basica.4y', 'secondary', 'university', 'unknown', 'basica.6y', 'professional course', 'without studies')
   - ```Default```: do you have credit in default? (categorical: "yes", "no", "unknown")
     - ```Home loan```: Do you have a home loan? (categorical: "yes", "no", "unknown")
     - ```Personal loan```: do you have a personal loan? (categorical: "yes", "no", "unknown")

2. Related to the contact of the last campaign:
   - ```Contact```: type of contact communication (categorical: "cellular", "phone")
   - ```Month```: month of last contact (categorical: "jan", "feb", "mar", ..., "nov", "dec")
   - ```Day of the week```: day of the week of the last contact of (categorical: "mon", "tue", "wed", "thu", "fri")

3. Other attributes:
   - ```Number of contacts```: number of contacts made with this client during this client campaign (numeric)
   - ```Days```: number of days that passed from the previous campaign to the current one (numeric; 999 means that the client was not previously contacted)
   - ```Previous number of contacts```: number of contacts made with this client during the previous campaign (numeric)
   - ```Previous result```: result of the previous marketing campaign (categorical: 'non-existent', 'failed', 'successful')

4. Attributes related to the social and economic context.
   - ```emp.var.rate```: employment variation rate - quarterly indicator (numeric)
   - ```cons.price.idx```: consumer price index - monthly indicator (numeric)
   - ```cons.conf.idx```: consumer confidence index - monthly indicator (numeric)
   - ```euribor3m```: 3-month interbank interest rate (acronym for Euro Interbank Offered Rate) - daily indicator (numeric)
   - ```nr.employees```: Quarterly average of the number of active citizens (numeric)

5. Output variable (target):
   - ```y``` - has the client taken a term deposit? (binary: "yes", "no")

## Cost-Profit Matrix
The cost of a telemarketing call made in a call center is approximately EUR 55 (it is calculated through a budget of 2.970.000 generated for 18 operators per month with 3.000 calls each, that is, a total of 54.000 calls of 1 minute average).

In the case of failed calls, the median call time is 8 minutes, giving a unit cost of EUR 440. These calls are shorter than successful ones, because most customers do not accept the product offered immediately. and others end up deciding to reject the proposal in subsequent calls.

In the case of a successful call (sale), the median contact time with the potential buyer, then the product buyer, is 20 minutes. To the total 1.100 associated with the cost of telemarketing (20 * EUR 55) we add 285 proration of miscellaneous expenses for advice, calculation and definition of the amount for the product offered, etc.
 
The average value of time deposits deposited by customers who were offered this promotional product in the past was EUR 100.000, leaving a final unit profit (after taxes) of EUR 4.000.

It is important to mention that the detection of clients who are not going to be called is usually rewarded because they are considered negative. This award is associated with different opportunity costs, both for the hiring/outsourcing of employees/Data Science consulting, as well as for the hiring of telemarketing companies and the consequent technical training of the call center team for attention according to the particularities and complexities of the situation, thus increasing the contractual costs with the outsourced company. For these reasons, true negatives are assigned 200.

| | Successful  call | Failed call
| - | :---: | :---: |
| Maded call | 2900 | -440 |
| Not made call | 0 | 200

## Metrics

The performance measure used to qualify the works will be: *Expected Utility*.

## Files description

```classification.py```: here you'll find the notebook with the final problem solution.
```optimization.py```: notebook used to tune XGBoost hyperparameters with Scikit-Learn.
