from django import forms

from predict.models import Product


class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = (
            'back_camera',
            'front_camera',
            'resolution_1',
            'resolution_2',
            'screen_size',
            'battery',
            'price',
            'pre_release_demand',
            # 'sales',
            # 'quarter',
        )

    def save(self, commit=True):
        product = super(ProductForm, self).save(commit=False)
        product.back_camera = self.cleaned_data['back_camera']
        product.front_camera = self.cleaned_data['front_camera']
        product.resolution_1 = self.cleaned_data['resolution_1']
        product.resolution_2 = self.cleaned_data['resolution_2']
        product.screen_size = self.cleaned_data['screen_size']
        product.battery = self.cleaned_data['battery']
        product.price = self.cleaned_data['price']
        product.pre_release_demand = self.cleaned_data['pre_release_demand']
        # product.sales = self.cleaned_data['sales']
        # product.quarter = self.cleaned_data['quarter']

        if commit:
            product.save()
        return product
