from django.shortcuts import render
from django.http import HttpResponse
from predict.forms import ProductForm

# Create your views here.
def index(request):
    args = {'name': 'Mrinal'}
    return render(request, 'predict/index.html', args)

def processing(request):
    pass

def product_describe_view(request):
    """
    View to take the data from the user and process year
    """
    product_added = False
    initiator = request.user
    if request.method == 'POST':
        print(20)
        product_form = ProductForm(data=request.POST)
        if product_form.is_valid():
            print(22)
            product_detail = product_form.save()
            product_added = True
            product_detail.save()
    # Not a HTTP POST, so we render our form using two ModelForm instances.
    # These forms will be blank, ready for user input.
    else:
        product_form = ProductForm()
    return render(request, 'predict/add_product.html',
        {'product_form': product_form, 'product_added': product_added})