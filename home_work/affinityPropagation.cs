using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace AffinityPropagation
{
    class Info
    {
        public Int32 vertexEnd { get; set; }

        public Int32 A { get; set; } = 0;
        public Int32 R { get; set; } = 0;
        public Int32 S { get; set; }

        public Info(Int32 v2, Int32 ss)
        {
            vertexEnd = v2;
            S = ss;
        }

    }

    class Program
    {
        const Int32 N = 196591;
        static Info[][] s_edges = new Info[N][];

        static IDictionary<Int32, Int32> clasters = new Dictionary<Int32, Int32>();

        static void ReadEdges()
        {
            using (StreamReader sr = new StreamReader(@"C:\Users\Marina\Documents\Visual Studio 2015\Projects\AffinityPropagation\AffinityPropagation\Gowalla_edges_modify.txt", System.Text.Encoding.Default))
            {
                string line;

                while ((line = sr.ReadLine()) != null)
                {
                    String[] vertex = line.Split(' ');
                    Int32 len = Convert.ToInt32(vertex[1]) + 1;
                    Int32 v = Convert.ToInt32(vertex[0]);
                    s_edges[v] = new Info[len];
                    for (int i = 2; i < len + 1; ++i)
                    {
                        s_edges[v][i - 2] = new AffinityPropagation.Info(Convert.ToInt32(vertex[i]), 1);
                    }
                }
            }
        }

        static void AddKK()
        {
            for (int i = 0; i < N; ++i)
            {
                s_edges[i][s_edges[i].Count() - 1] = new Info(i, -1);                
            }
        }

        //static void makeNewDataFormat()
        //{
        //    using (StreamWriter sw = new StreamWriter(@"C:\Users\Marina\Documents\Visual Studio 2015\Projects\AffinityPropagation\AffinityPropagation\Gowalla_edges_modify.txt", true, System.Text.Encoding.Default)) { 

        //        foreach (var a in s_edges)
        //        {
        //            sw.Write(a.Key + " " + a.Value.Count);
        //            foreach(var b in a.Value)
        //            {
        //                sw.Write(" " + b.vertexEnd);
        //            }
        //            sw.WriteLine();
        //        }
        //    }
        //}

        static Int32 FindMax(Int32 v1, Int32 v2) // v1, v2 -- vertex
        {
            Int32 sumMax = Int32.MinValue;

            for (int k = 0; k < s_edges[v1].Count(); ++k)
            {
                if (s_edges[v1][k].vertexEnd == v2)
                    continue;
                if (sumMax < s_edges[v1][k].A + s_edges[v1][k].S)
                    sumMax = s_edges[v1][k].A + s_edges[v1][k].S;
            }
            return sumMax;
        }

        static Info KK(Int32 k)
        {
            for (int i = 0; i < s_edges[k].Count(); ++i)
            {
                if (s_edges[k][i].vertexEnd == k)
                    return s_edges[k][i];
            }
            return null;
        }


        static Int32 ArgMax(Int32 v1)
        {
            Int32 index = 0;
            Int32 max = Int32.MinValue;
            foreach (Info k in s_edges[v1])
            {
                Info edge = s_edges[v1].Where(p => p.vertexEnd == k.vertexEnd).First();
                if (edge.R + edge.A > max)
                {
                    max = edge.R + edge.A;
                    index = k.vertexEnd;
                }
            }

            return index;
        }

        static Int32 FindSumMax(Int32 i, Int32 k) // i -- vertex, k -- index
        {
            Int32 sum = 0;
            Int32 kv = s_edges[i][k].vertexEnd;

            for (int j = 0; j < s_edges[kv].Count(); ++j)
            {
                Int32 jv = s_edges[kv][j].vertexEnd;

                if (kv == jv || jv == i)
                    continue;

                sum += Math.Max(0, s_edges[kv][j].R);
            }
            return sum;
        }


        static void Main(string[] args)
        {
            ReadEdges();
            //  makeNewDataFormat();
            AddKK();

            for (int iter = 0; iter < 20; ++iter)
            {
                for (int i = 0; i < N; ++i) { 
                    for (int k = 0; k < s_edges[i].Count(); ++k)
                    {
                        s_edges[i][k].R = s_edges[i][k].S - FindMax(i,k);

                        if (i != s_edges[i][k].vertexEnd)
                        {
                            s_edges[i][k].A = Math.Min(0, KK(k).R + FindSumMax(i, k));
                        }
                        else
                        {
                            s_edges[i][k].A = FindSumMax(i,k);
                        }
                    }
                }
            }

            ISet<Int32> res = new HashSet<Int32>();

            for (Int32 i = 0; i < N; ++i)
            {
                clasters.Add(i, ArgMax(i));
                res.Add(ArgMax(i));
            }


            //foreach (var a in clasters)
            //{
            //    Console.WriteLine("vertex: " + a.Key + " claster: " + a.Value);
            //}
            Console.WriteLine(res.Count());
        }

    }
}
