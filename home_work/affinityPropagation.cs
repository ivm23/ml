using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace affinityPropagation
{
    class Edge
    {
        public Int32 vertexStart { get; set; }
        public Int32 vertexEnd { get; set; }

        public Int32 A { get; set; } = 0;
        public Int32 R { get; set; } = 0;

        public Edge(Int32 v1, Int32 v2)
        {
            vertexStart = v1;
            vertexEnd = v2;
        }
    }

    class Program
    {
        static IList<Edge> s_edges = new List<Edge>();
        static IDictionary<Int32, Int32> clasters = new Dictionary<Int32, Int32>();

        static public ISet<Int32> GetAllVertex()
        {
            ISet<Int32> allVertex = new HashSet<Int32>();
            var a = s_edges.Select(p => p.vertexStart).Union(s_edges.Select(p => p.vertexEnd));
            foreach (var b in a)
            {
                allVertex.Add(b);
            }
            return allVertex;
        }

        static void ReadEdges()
        {
            using (StreamReader sr = new StreamReader(@"C:\Users\Ivanova.M\Documents\Visual Studio 2015\Projects\affinityPropagation\affinityPropagation\Gowalla_edges.txt", System.Text.Encoding.Default))
            {
                string line;
                
                while ((line = sr.ReadLine()) != null)
                {
                    String[] vertex = line.Split('	');
                    s_edges.Add(new Edge(Convert.ToInt32(vertex[0]), Convert.ToInt32(vertex[1])));
                }
            }
        }

        static void AddKK()
        {
            foreach (var a in GetAllVertex())
            {
                s_edges.Add(new Edge(a, a));
            }
        }
        static Int32 FindMax(Int32 v1, Int32 v2)
        {
            return s_edges.Where(p => (p.vertexStart == v1 && p.vertexEnd != v2)).Max(p => p.A + 1);
        }

        static Int32 FindSumMax(Int32 v1, Int32 v2)
        {
            IEnumerable<Edge> edges = s_edges.Where(p => (p.vertexStart == v1 && p.vertexEnd != v2));
            Int32 sum = 0;
            foreach (Edge edge in edges)
            {
                sum += Math.Max(0, edge.R);
            }
            return sum;
        }

        static Int32 FindMinSum(Int32 v1, Int32 v2)
        {
            Int32 sum = FindSumMax(v1, v2);
            return Math.Min(0, (s_edges.Where(p => (p.vertexStart == v2 && p.vertexEnd == v2)).First()).R + sum);
        }

        static Int32 ArgMax(Int32 v1)
        {
            Int32 index = 0;
            Int32 max = Int32.MinValue;
            foreach(Int32 k in GetAllVertex())
            {
                Edge edge = s_edges.Where(p => (p.vertexStart == v1 && p.vertexEnd == k)).First();
                if (edge.R + edge.A > max)
                {
                    max = edge.R + edge.A;
                    index = k;
                }
            }

            return index;
        }
        static void Main(string[] args)
        {
            ReadEdges();
            AddKK();
            for (int i = 0; i < 1; ++i)
            {
                foreach (Edge edge in s_edges)
                {
                    edge.R = 1 - FindMax(edge.vertexStart, edge.vertexEnd);
                    if (edge.vertexStart != edge.vertexEnd)
                    {
                        edge.A = FindSumMax(edge.vertexStart, edge.vertexEnd);
                    }
                    else
                    {
                        edge.A = Math.Min(0, FindSumMax(edge.vertexEnd, edge.vertexEnd));
                    }
                }
                foreach(Int32 v in GetAllVertex())
                {
                    clasters.Add(v, ArgMax(v));
                }
            }

            foreach(var a in clasters)
            {
                Console.WriteLine("vertex: " + a.Key + " claster: " + a.Value);
            }
        }
    }
}
