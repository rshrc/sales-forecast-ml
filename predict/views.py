from django.shortcuts import render

from ml_code.ml_process import server_predictor
from predict.forms import ProductForm


def operate_function(product_detail):
    back_camera = product_detail.back_camera
    front_camera = product_detail.front_camera
    resolution_1 = product_detail.resolution_1
    resolution_2 = product_detail.resolution_2
    screen_size = product_detail.screen_size
    battery = product_detail.battery
    price = product_detail.price
    pre_release_demand = product_detail.pre_release_demand
    # sales = product_detail.sales
    # quarter = product_detail.quarter
    cluster_assigned, predicted_sales = server_predictor.get_prediction(back_camera, front_camera, resolution_1,
                                                                        resolution_2, screen_size,
                                                                        battery, price, pre_release_demand
                                                                        )
    return cluster_assigned[0], int(predicted_sales[0])


def product_describe_view(request):
    """
    View to take the data from the user and process year
    """
    product_added = False
    cluster_assigned = 0
    predicted_sales = 0
    if request.method == 'POST':
        product_form = ProductForm(data=request.POST)
        if product_form.is_valid():
            product_detail = product_form.save()
            product_added = True
            print(product_detail)
            cluster_assigned, predicted_sales = operate_function(product_detail)
            print("Cluster Assigned : {}".format(cluster_assigned))
            product_detail.save()
    # Not a HTTP POST, so we render our form using two ModelForm instances.
    # These forms will be blank, ready for user input.
    else:
        product_form = ProductForm()
    return render(request, 'predict/add_product.html',
                  {'product_form': product_form, 'product_added': product_added, 'cluster_assigned': cluster_assigned,
                   'predicted_sales': predicted_sales})
