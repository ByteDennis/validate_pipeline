from setuptools import setup, find_packages

setup(
    name="data-migration-pipeline",
    version="1.0.0",
    description="PCDS to AWS Data Migration Pipeline",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "awswrangler>=3.0.0",
        "boto3>=1.26.0",
        "pyarrow>=10.0.0",
        "oracledb>=1.3.0",
        "pyathena>=3.0.0",
        "ldap3>=2.9.0",
        "loguru>=0.6.0",
        "confection>=0.1.0",
        "tqdm>=4.64.0",
        "openpyxl>=3.0.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
)