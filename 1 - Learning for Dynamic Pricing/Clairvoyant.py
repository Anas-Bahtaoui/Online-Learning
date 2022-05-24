from parameters import product_configs, products, environment, purchase_amounts, customer_counts
from Customer import Customer

# Function that computes the clairvoant reward and the optimal price configuration
def clairvoyant_reward(products, purchase_amounts, customer_counts):    
    reward_vector = []
   
    for candite_price_1 in products[0].get_candidate_prices():
        print("Candidate price 1:", candite_price_1)
        
        for candite_price_2 in products[1].get_candidate_prices():
            print("Candidate price 2:", candite_price_2)
            
            for candite_price_3 in products[2].get_candidate_prices():
                print("Candidate price 3:", candite_price_3)
                
                for candite_price_4 in products[3].get_candidate_prices():
                    print("Candidate price 4:", candite_price_4)
                    
                    for candite_price_5 in products[4].get_candidate_prices():
                        print("Candidate price 5:", candite_price_5)
                        
                        # Calculate the reward
                        reward = 0
                        #for class_ in list(CustomerClass):
                            
                            # TODO : Do we want to sort out the configurations that are impossible (reservation_price < candite_price) => Only keep the configurations that are beeing bought
                            # TODO : OR we use the conversionrate = #Customer * ConversionRate
                            
                            # reservation_price = reservation_price_distribution_from_curves(class_, 0, candite_price_1).get_expectation()
                            # is_purchased = reservation_price >= candite_price_1
                            # if not is_purchased:
                            #     continue
                            # print("Purchased, reservation price:", reservation_price, "product price:", candite_price_1)
                            # expected_purchase_count = purchase_amounts[class_][0].get_expectation()
                            # result_ = candite_price_1 * customer_counts[class_].get_expectation()
        
clairvoyant_reward(products, purchase_amounts, customer_counts)