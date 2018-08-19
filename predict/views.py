from django.shortcuts import render
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
    sales = product_detail.sales
    quarter = product_detail.quarter
    return back_camera + front_camera + resolution_1 + resolution_2 + screen_size + battery + price + pre_release_demand + sales + quarter


def product_describe_view(request):
    """
    View to take the data from the user and process year
    """
    product_added = False
    result = 0
    if request.method == 'POST':
        product_form = ProductForm(data=request.POST)
        if product_form.is_valid():
            product_detail = product_form.save()
            product_added = True
            print(product_detail)
            result = operate_function(product_detail)
            print("Back Camera = {}".format(result))
            product_detail.save()
    # Not a HTTP POST, so we render our form using two ModelForm instances.
    # These forms will be blank, ready for user input.
    else:
        product_form = ProductForm()
    return render(request, 'predict/add_product.html',
                  {'product_form': product_form, 'product_added': product_added, 'result': result})
