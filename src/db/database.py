#This module instantiates the database connection
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

#sqlite for development, PostgresSQL for production
DATABASE_URL="sqlite:///./skill_tree.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base =  declarative_base()